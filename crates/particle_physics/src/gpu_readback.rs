use crate::PARTICLE_NUM;
use bevy::{
    app::{App, Plugin},
    asset::AssetServer,
    ecs::{
        entity::Entity,
        schedule::{
            apply_deferred,
            common_conditions::{any_with_component, not, resource_exists},
            IntoSystemConfigs,
        },
        system::{Commands, Query, Res, ResMut, Resource},
        world::{FromWorld, World},
    },
    math::Vec2,
    render::{
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::storage_buffer, encase, BindGroup, BindGroupEntries, BindGroupLayout,
            BindGroupLayoutEntries, Buffer, BufferAddress, BufferDescriptor, BufferInitDescriptor,
            BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, Maintain, MapMode, PipelineCache, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Extract, ExtractSchedule, Render, RenderApp, RenderSet,
    },
    utils::synccell::SyncCell,
};
use std::{
    sync::mpsc::{channel, Receiver, Sender},
    time::Instant,
};

// The length of the buffer sent to the gpu
const BUFFER_LEN: usize = u16::MAX as _;

/// This will receive asynchronously any data sent from the render world
#[derive(Resource)]
pub struct MainWorldReceiver(SyncCell<Receiver<Vec<Vec2>>>);

/// This will send asynchronously any data to the main world
#[derive(Resource)]
struct RenderWorldSender(Sender<Vec<Vec2>>);

/// Label to identify the node in the render graph
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ComputeNodeLabel;

/// Handle to the compute pipeline and its correspdoning bindgroup layout
#[derive(Debug, Resource)]
struct ComputePipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}
impl FromWorld for ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("compute bind group layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                [
                    // u32::MAX is used because the idx is ignored when using `sequential` and must be set to u32::MAX
                    storage_buffer::<Vec<Vec2>>(false).build(u32::MAX, ShaderStages::COMPUTE), // accel
                    storage_buffer::<Vec<Vec2>>(false).build(u32::MAX, ShaderStages::COMPUTE), // position
                    storage_buffer::<Vec<f32>>(false).build(u32::MAX, ShaderStages::COMPUTE), // mass
                    storage_buffer::<u32>(false).build(u32::MAX, ShaderStages::COMPUTE),      // len
                ],
            ),
        );

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/gpu_readback.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("GPU readback compute shader".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: Vec::new(),
            entry_point: "main".into(),
        });
        ComputePipeline { layout, pipeline }
    }
}

/// Compute shader bindgroup (instance resources bound that with layout given in BindGroupLayout)
#[derive(Resource)]
struct GpuBufferBindGroup(BindGroup);
fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputePipeline>,
    render_device: Res<RenderDevice>,
    buffers: Res<Buffers>,
) {
    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::sequential((
            buffers.gpu_buffer.as_entire_binding(),
            buffers.positions.as_entire_binding(),
            buffers.masses.as_entire_binding(),
            buffers.len.as_entire_binding(),
        )),
    );
    commands.insert_resource(GpuBufferBindGroup(bind_group));
}

fn map_and_read_buffer(
    render_device: Res<RenderDevice>,
    buffers: Res<Buffers>,
    sender: Res<RenderWorldSender>,
) {
    // First we get a buffer slice which represents a chunk of the buffer (which we
    // can't access yet).
    let buffer_slice = buffers.cpu_buffer.slice(..);

    let (s, r) = channel::<()>();

    // Maps the buffer so it can be read on the cpu
    let map_async = Instant::now();
    buffer_slice.map_async(MapMode::Read, move |r| match r {
        // This will execute once the gpu is ready, so after the call to poll()
        Ok(_) => s.send(()).expect("Failed to send map update"),
        Err(err) => panic!("Failed to map buffer: {err}"),
    });

    // Need to poll or queue.submit for map_async to finish.
    // On web poll happens automatically and this is noop.
    // On desktop this polls until all jobs submitted so far have finished and so map_async will have finished.
    // TODO panic_on_timeout is a noop?
    render_device.poll(Maintain::Wait).panic_on_timeout();
    dbg!(map_async.elapsed()); // TODO does map_async block itself on desktop?

    // This blocks until the buffer is mapped
    r.recv().expect("Failed to receive the map_async message");

    {
        let buffer_view = buffer_slice.get_mapped_range();
        let data = buffer_view
            .chunks(std::mem::size_of::<Vec2>())
            .map(|chunk| {
                Vec2::from_array([
                    f32::from_ne_bytes(chunk[0..4].try_into().unwrap()),
                    f32::from_ne_bytes(chunk[4..8].try_into().unwrap()),
                ])
            })
            .collect::<Vec<Vec2>>();
        // If receive end dropped first then likely shutting down (not a bug)
        let _ = sender.0.send(data);
    }

    // We need to make sure all `BufferView`'s are dropped before we do what we're about
    // to do.
    // Unmap so that we can copy to the staging buffer in the next iteration.
    buffers.cpu_buffer.unmap();
}

/// This system will block until to get the data sent from the render world
pub fn receive(mut receiver: ResMut<MainWorldReceiver>) -> Option<std::vec::Vec<Vec2>> {
    receiver.0.get().try_recv().ok()
}

// We need a plugin to organize all the systems and render node required for this example
pub struct GpuReadbackPlugin;
impl Plugin for GpuReadbackPlugin {
    fn build(&self, _app: &mut App) {}

    // The render device is only accessible inside finish().
    // So we need to initialize render resources here.
    fn finish(&self, app: &mut App) {
        let (s, r) = channel();
        app.insert_resource(MainWorldReceiver(SyncCell::new(r)));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(RenderWorldSender(s))
            .init_resource::<ComputePipeline>()
            // .init_resource::<Buffers>()
            .add_systems(
                Render,
                (
                    prepare_bind_group
                        .in_set(RenderSet::PrepareBindGroups)
                        // We don't need to recreate the bind group every frame
                        .run_if(not(resource_exists::<GpuBufferBindGroup>)),
                    // We need to run it after the render graph is done
                    // because this needs to happen after submit()
                    map_and_read_buffer.after(RenderSet::Render),
                )
                    .run_if(resource_exists::<Buffers>),
            )
            .add_systems(
                ExtractSchedule,
                (
                    |mut commands: Commands,
                    query: Extract<Query<(Entity, &crate::Position, &crate::Mass)>>| {
                        query.iter().for_each(|(entity, position, mass)| {
                            commands
                                .get_or_spawn(entity)
                                .insert((position.clone(), mass.clone()));
                        });
                    },
                    apply_deferred,
                    (|world: &mut World| {
                        world.init_resource::<Buffers>();
                    })
                    .run_if(not(resource_exists::<Buffers>))
                    .run_if(any_with_component::<crate::Position>)
                    .run_if(any_with_component::<crate::Mass>),
                    update_input_buffers,
                ).chain(),
            );

        // Add the compute node as a top level node to the render graph
        // This means it will only execute once per frame
        render_app
            .world
            .resource_mut::<RenderGraph>()
            .add_node(ComputeNodeLabel, ComputeNode::default());
    }
}

#[derive(Resource)]
struct Buffers {
    /// Group 1
    positions: Buffer,
    /// Group 2
    masses: Buffer,
    // The buffer that will be used by the compute shader to save the output.
    /// Group 0
    gpu_buffer: Buffer,
    // The buffer that will be read on the cpu.
    // The `gpu_buffer` will be copied to this buffer every frame
    cpu_buffer: Buffer,
    /// Number of valid particles
    // Group 3
    len: Buffer,
}

impl FromWorld for Buffers {
    fn from_world(world: &mut World) -> Self {
        let (positions, masses): (Vec<Vec2>, Vec<f32>) = world
            .query::<(&crate::components::Position, &crate::Mass)>()
            .iter(world)
            .map(|(&x, &y)| (x.0, y.0))
            .unzip();
        let len = u32::try_from(positions.len()).unwrap();

        let render_device = world.resource::<RenderDevice>();
        let mut init_data = encase::StorageBuffer::new(Vec::new());
        // Init the buffer
        let data = vec![Vec2::splat(f32::NAN); BUFFER_LEN];
        init_data.write(&data).expect("Failed to write buffer");
        // The buffer that will be accessed by the gpu
        let gpu_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gpu_buffer"),
            contents: init_data.as_ref(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        // The buffer that will be accessed by the cpu
        // Have to synchronize with [`CommandEncoder::copy_buffer_to_buffer`]
        let cpu_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("readback_buffer"),
            size: (BUFFER_LEN * std::mem::size_of::<Vec2>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let masses = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("masses"),
            contents: {
                let mut out = encase::StorageBuffer::new(Vec::new());
                out.write(&masses).expect("Failed to write buffer");
                out
            }
            .as_ref(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let positions = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("positions"),
            contents: {
                let mut out = encase::StorageBuffer::new(Vec::new());
                out.write(&positions).expect("Failed to write buffer");
                out
            }
            .as_ref(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let len = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("len"),
            contents: {
                let mut out = encase::StorageBuffer::new(Vec::new());
                out.write(&len).expect("Failed to write buffer");
                out
            }
            .as_ref(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        Self {
            gpu_buffer,
            cpu_buffer,
            masses,
            positions,
            len,
        }
    }
}

/// Write new values to input buffers using the [`RenderQueue`] and `write_buffer` before shader does computation.
fn update_input_buffers(
    queue: ResMut<RenderQueue>,
    buffers: Res<Buffers>,
    query: Query<(&crate::Position, &crate::Mass)>,
) {
    let (positions, masses): (Vec<Vec2>, Vec<f32>) =
        query.iter().map(|(&x, &y)| (x.0, y.0)).unzip();
    let len = positions.len();
    let masses = {
        let mut out = encase::StorageBuffer::new(Vec::new());
        out.write(&masses).expect("Failed to write buffer");
        out
    };
    let positions = {
        let mut out = encase::StorageBuffer::new(Vec::new());
        out.write(&positions).expect("Failed to write buffer");
        out
    };
    queue.0.write_buffer(&buffers.masses, 0, masses.as_ref());
    queue
        .0
        .write_buffer(&buffers.positions, 0, positions.as_ref());
    queue.0.write_buffer(
        &buffers.len,
        0,
        u32::try_from(len).unwrap().to_ne_bytes().as_slice(),
    )
}

/// The node that will execute the compute shader
#[derive(Default)]
struct ComputeNode {}
impl render_graph::Node for ComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputePipeline>();
        let Some(bind_group) = world.get_resource::<GpuBufferBindGroup>() else {
            return Ok(());
        };

        // If the pipeline doesn't exist that means the render started before the pipeline was created and the shader can't run.
        let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
            return Ok(());
        };
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("GPU readback compute pass"),
                    ..Default::default()
                });

        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.set_pipeline(init_pipeline);
        pass.dispatch_workgroups(((PARTICLE_NUM as u32 - 1) / 64) + 1, 1, 1);
        drop(pass);

        // Copy the gpu accessible buffer to the cpu accessible buffer
        let buffers = world.resource::<Buffers>();
        render_context.command_encoder().copy_buffer_to_buffer(
            &buffers.gpu_buffer,
            0,
            &buffers.cpu_buffer,
            0,
            (BUFFER_LEN * std::mem::size_of::<Vec2>()) as BufferAddress,
        );

        Ok(())
    }
}
