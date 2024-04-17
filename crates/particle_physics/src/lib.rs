use bevy::{
    app::{Plugin, Startup, Update},
    ecs::schedule::{IntoSystemConfigs, ScheduleLabel},
    prelude::*,
    utils::HashSet,
};
use components::{Acceleration, Mass, NewAcceleration, Particle, Position, Radius, Velocity};
use spatial_hash::SparseGrid2d;
use std::marker::PhantomData;

pub mod components;

/// Gravitational constant
const G: f32 = 6.67430e-11;

#[derive(Default, Debug, Clone, Copy)]
struct Energy(f32);

struct Kinetic;
struct Potential;

#[derive(Resource)]
struct GraphPoints<Kind, T> {
    points: Vec<T>,
    _ty: PhantomData<Kind>,
}
impl<K, T: std::fmt::Debug> std::fmt::Debug for GraphPoints<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphPoints")
            .field("points", &self.points)
            .field("_ty", &self._ty)
            .finish()
    }
}
impl<K, T> Default for GraphPoints<K, T> {
    fn default() -> Self {
        Self {
            points: Default::default(),
            _ty: Default::default(),
        }
    }
}
impl<K, T: Clone> Clone for GraphPoints<K, T> {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            _ty: self._ty,
        }
    }
}

#[derive(Debug, Default, Resource)]
struct SpatialHashes(pub Vec<SparseGrid2d>);

/// Move sprites to entities' position. Assumes the window is [-1.,1.] in both x and y
fn sync_position_and_sprite(mut query: Query<(&mut Transform, &Position), Changed<Position>>) {
    query.par_iter_mut().for_each(|(mut transform, position)| {
        transform.translation = Vec3::new(position.0.x, position.0.y, 0.)
    });
}
/// Sync sprite with size
fn sync_radius_and_sprite(
    mut meshes: ResMut<Assets<Mesh>>,
    mut particles: Query<(&mut bevy::sprite::Mesh2dHandle, &Radius), Changed<Radius>>,
) {
    particles.iter_mut().for_each(|(mut sprite, radius)| {
        *sprite = meshes.add(Circle::new(radius.0)).into();
    })
}

/// Generates spatial hashes with resolutions depending on radii
fn gen_spatial_hashes(mut hashes: ResMut<SpatialHashes>, query: Query<&Radius>) {
    if hashes.0.is_empty() {
        hashes
            .0
            .push(SparseGrid2d::new(2. * query.iter().next().unwrap().0))
    }
}

/// Rebuild spatial hashes and insert [`Position`]s into spatial hashes
fn update_spatial_hashes(mut hashes: ResMut<SpatialHashes>, query: Query<(Entity, &Position)>) {
    hashes.0.iter_mut().for_each(|x| x.clear());
    query.iter().for_each(|(entity, pos)| {
        for hash in hashes.0.iter_mut() {
            hash.insert_point(pos.0, entity)
        }
    })
}

/// Update position with verlet integration.
fn update_position(mut query: Query<(&mut Position, &Velocity, &NewAcceleration)>) {
    query
        .par_iter_mut()
        .for_each(|(mut position, velocity, acceleration)| {
            // position += timestep * (velocity + timestep * acceleration / 2);
            // timestep = 1
            position.0 += velocity.0 + acceleration.0 / 2.
        });
}
/// Update velocity with verlet integration.
fn update_velocity(mut query: Query<(&mut Velocity, &Acceleration, &NewAcceleration)>) {
    query
        .par_iter_mut()
        .for_each(|(mut velocity, acceleration, new_acceleration)| {
            // velocity += timestep * (acceleration + newAcceleration) / 2;
            // timestep = 1
            velocity.0 += (acceleration.0 + new_acceleration.0) / 2.;
        });
}
/// Calculate acceleration from force, and move `NewAcceleration` to `Acceleration`
fn update_acceleration(
    mut accelerators: Query<
        (Entity, &Position, &mut Acceleration, &mut NewAcceleration),
        With<Mass>,
    >,
    attractors: Query<(Entity, &Position, &Mass)>,
) {
    /// Acceleration due to gravity towards other
    fn attraction_to(this: &Position, other: (&Position, &Mass)) -> NewAcceleration {
        let dp = other.0 .0 - this.0;
        let dp_len = dp.length();
        // Acceleration(dp / dp_len * G * other.1 .0 / dp_len.powi(2))
        // perf: simplification of above, self mass cancels when calculating acceleration
        NewAcceleration(dp * other.1 .0 / dp_len.powi(3))
    }

    // newAcceleration = force(time, position) / mass;
    accelerators.par_iter_mut().for_each(
        |(outer_entity, outer_position, mut accel, mut new_accel)| {
            accel.0 = new_accel.0; // on new loop new_accel becomes old
            new_accel.0 = Vec2::ZERO;
            attractors
                .iter()
                .filter(|(inner_entity, _, _)| inner_entity != &outer_entity)
                .for_each(|(_, position, mass)| {
                    new_accel.0 += attraction_to(outer_position, (position, mass)).0
                });
            new_accel.0 *= G; // distributive law
        },
    );
}
/// Set the size of objects proportional to their mass
fn sync_mass_and_radius(mut particles: Query<(&mut Radius, &Mass), Changed<Mass>>) {
    particles
        .par_iter_mut()
        .for_each(|(mut radius, mass)| radius.0 = mass.0.sqrt() / 1000.);
}

/// Merge entities that collide
fn merge_colliders(
    mut commands: Commands,
    mut shared_query: Query<(&mut Mass, Option<&mut Velocity>)>,
    merge_target: Query<(Entity, &Position, &Radius), With<Mass>>,
    merge_source: Query<(Entity, &Position), With<Mass>>,
    hashes: Res<SpatialHashes>,
) {
    let hash = hashes.0.first().expect("missing spatial hash");
    let mut to_despawn = HashSet::new();
    // Without grouping all sources at once this won't be deterministic in parallel because floats don't commute
    merge_target.iter().for_each(|target| {
        // iterate over items within the radius of self
        hash.aabb_iter(Rect::from_center_half_size(
            target.1 .0,
            Vec2::splat(target.2 .0),
        ))
        .map(|entity| merge_source.get(entity).unwrap())
        .filter(|source| {
            source.0 != target.0
                    // source in target
                    && (target.1.0.distance(source.1.0)) < (target.2.0)
        }) // keep entities that are not the same and contains in the target
        .for_each(|source| {
            // source smaller than target
            let target_mass = *shared_query.get(target.0).unwrap().0;
            let source_mass = *shared_query.get(source.0).unwrap().0;
            // only despawn smaller
            if target_mass.0 > source_mass.0
                // or first encountered of same mass
                    || (target_mass.0 == source_mass.0 && !to_despawn.contains(&target.0))
            {
                let mass_final = Mass(source_mass.0 + target_mass.0);
                // Merge velocity with momentum conservation
                if let Some(source_velocity) = shared_query.get(source.0).unwrap().1.cloned() {
                    if let Some(mut target_velocity) = shared_query.get_mut(target.0).unwrap().1 {
                        target_velocity.0 = (source_mass.0 * source_velocity.0
                            + target_mass.0 * target_velocity.0)
                            / mass_final.0;
                    }
                }

                // Combine mass
                *shared_query.get_mut(target.0).unwrap().0 = mass_final;
                to_despawn.insert(source.0);
            }
        })
    });
    // Despawn queued
    to_despawn
        .into_iter()
        .for_each(|x| commands.entity(x).despawn());
}

fn setup(
    mut commands: Commands,
    window: Query<&Window>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let window = window.single();
    // Camera
    commands.spawn(Camera2dBundle {
        transform: Transform {
            scale: Vec3::ONE / window.height().min(window.width()),
            ..Default::default()
        },
        ..Default::default()
    });

    let material = materials.add(ColorMaterial::from(Color::RED));
    // Randomized initial particles
    commands.spawn_batch(
        std::iter::repeat_with(move || {
            Particle::new_rand(
                Rect::new(-0.4, -0.4, 0.4, 0.4),
                Rect::new(-0.001, -0.001, 0.001, 0.001),
                material.clone(),
            )
        })
        .take(10000),
    );

    commands.spawn(Particle {
        position: Position(Vec2::new(0.25 / 8., 0.)),
        velocity: Velocity(Vec2::new(0., -0.0002)),
        mass: Mass(100.),
        ..Default::default()
    });
    commands.spawn(Particle {
        position: Position(Vec2::new(-0.25 / 8., 0.)),
        velocity: Velocity(Vec2::new(0., 0.0002)),
        mass: Mass(100.),
        ..Default::default()
    });
    commands.spawn(Particle {
        position: Position(Vec2::new(0.25 / 8. + 0.05, 0.)),
        velocity: Velocity(Vec2::new(0., -0.00040)),
        mass: Mass(10.),
        ..Default::default()
    });
    commands.spawn(Particle {
        position: Position(Vec2::new(-0.25 / 8. - 0.05, 0.)),
        velocity: Velocity(Vec2::new(0., 0.00040)),
        mass: Mass(10.),
        ..Default::default()
    });
}

fn kinetic_energy(
    query: Query<(&Velocity, &Mass)>,
    mut energy: ResMut<GraphPoints<Kinetic, Energy>>,
) {
    let val = query
        .iter()
        .map(|(Velocity(v), Mass(m))| 0.5 * m * v.length().powi(2))
        .sum::<f32>();
    energy.points.push(Energy(val));
    // println!("K energy: {:?}", val);
}

fn potential_energy(
    query: Query<(Entity, &Position, &Mass)>,
    query2: Query<(Entity, &Position, &Mass)>,
    mut energy: ResMut<GraphPoints<Potential, Energy>>,
) {
    // perf: found Mutex slightly faster than using std::sync::mpsc and equal to atomic_float::AtomicF32
    let acc = std::sync::Mutex::new(0.);
    query.par_iter().for_each(|x| {
        let val = query2
            .iter()
            .filter(move |y| x.0 != y.0)
            .map(|y| -(x.2 .0 * y.2 .0 * G) / x.1 .0.distance(y.1 .0))
            .sum::<f32>();
        *acc.lock().unwrap() += val;
    });
    let val = acc.lock().unwrap();
    energy.points.push(Energy(*val));
    // println!("P energy: {:?}", val);
}
fn graph(
    mut contexts: bevy_egui::EguiContexts,
    kinetic_energy: Res<GraphPoints<Kinetic, Energy>>,
    potential_energy: Res<GraphPoints<Potential, Energy>>,
) {
    use bevy_egui::egui;
    use egui_plot::{Line, Plot, PlotPoints};
    egui::Window::new("System Energy").show(contexts.ctx_mut(), |ui| {
        let vec_to_line = |vec: &Vec<_>| {
            Line::new(
                vec.iter()
                    .enumerate()
                    .map(|(i, val): (_, &Energy)| [i as f64, val.0 as f64])
                    .collect::<PlotPoints>(),
            )
        };
        let kinetic_line = vec_to_line(&kinetic_energy.points);
        let potential_line = vec_to_line(&potential_energy.points);
        ui.columns(2, |columns| {
            columns[0].label("Kinetic Energy");
            Plot::new("kinetic").show(&mut columns[0], |plot_ui| plot_ui.line(kinetic_line));
            columns[1].label("Potential Energy");
            Plot::new("potential").show(&mut columns[1], |plot_ui| plot_ui.line(potential_line));
        });
    });
}

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
/// [`Schedule`] of things to do when stepping the physics simulation.
pub struct PhysicsStep;

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Times to run [`PhysicsStep`] per frame.
pub struct Substeps(pub u32);
impl Default for Substeps {
    fn default() -> Self {
        Self(1)
    }
}

#[derive(Debug, Default)]
pub struct ParticlePhysicsPlugin {
    /// Graph statistics of simulation.
    pub graph: bool,
    /// Number of [`PhysicsStep`] to perform per frame.
    // TODO enum for adaptive to the framerate
    pub substeps: Substeps,
}

/// Run [`PhysicsStep`] [`Substeps`] number of times.
fn run_physics_step(world: &mut World) {
    for _ in 0..world
        .get_resource::<Substeps>()
        .copied()
        .unwrap_or_default()
        .0
    {
        world.run_schedule(PhysicsStep);
    }
}

impl Plugin for ParticlePhysicsPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let mut physics_step = Schedule::new(PhysicsStep);
        physics_step.set_build_settings(bevy::ecs::schedule::ScheduleBuildSettings {
            ambiguity_detection: bevy::ecs::schedule::LogLevel::Warn,
            ..Default::default()
        });
        app.add_schedule(physics_step)
            .add_systems(PostUpdate, run_physics_step); // TODO where to put this?

        app.add_systems(Startup, setup)
            .insert_resource(SpatialHashes::default())
            .insert_resource(self.substeps)
            .add_systems(
                PhysicsStep,
                (
                    (gen_spatial_hashes, update_spatial_hashes).chain(),
                    /*
                    Verlet integration see: <https://gamedev.stackexchange.com/questions/15708/how-can-i-implement-gravity/41917#41917>
                    acceleration = force(time, position) / mass;
                    time += timestep;
                    position += timestep * (velocity + timestep * acceleration / 2);
                    newAcceleration = force(time, position) / mass;
                    velocity += timestep * (acceleration + newAcceleration) / 2;
                    */
                    update_acceleration,
                    update_velocity,
                    update_position,
                    merge_colliders,
                )
                    .chain(),
            )
            .add_systems(Update, sync_mass_and_radius.after(merge_colliders))
            .add_systems(Update, sync_radius_and_sprite.after(sync_mass_and_radius))
            .add_systems(Update, sync_position_and_sprite.after(update_position));
        if self.graph {
            if !app.is_plugin_added::<bevy_egui::EguiPlugin>() {
                app.add_plugins(bevy_egui::EguiPlugin);
            }
            app.insert_resource(GraphPoints::<Kinetic, Energy>::default())
                .insert_resource(GraphPoints::<Potential, Energy>::default())
                .add_systems(PhysicsStep, kinetic_energy.after(merge_colliders))
                .add_systems(PhysicsStep, potential_energy.after(merge_colliders))
                .add_systems(Update, graph);
        }
    }
}
