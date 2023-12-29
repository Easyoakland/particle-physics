use bevy::{
    app::{App, Startup, Update},
    asset::{Assets, Handle},
    core_pipeline::core_2d::Camera2dBundle,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::{
        bundle::Bundle,
        component::Component,
        entity::Entity,
        event::Event,
        query::{Changed, With},
        schedule::IntoSystemConfigs,
        system::{Commands, Query, Res, ResMut, Resource},
    },
    math::{Rect, Vec2, Vec3},
    render::{
        color::Color,
        mesh::{shape, Mesh},
    },
    sprite::{ColorMaterial, MaterialMesh2dBundle, Mesh2dHandle},
    transform::components::Transform,
    utils::hashbrown::HashSet,
    window::Window,
    DefaultPlugins,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use rand::{thread_rng, Rng};
use std::sync::Mutex;

/// Gravitational constant
const G: f32 = 6.67430e-11;

#[derive(Debug, Clone, Default, Component)]
struct Position(Vec2);

#[derive(Debug, Clone, Default, Component)]
struct Velocity(Vec2);

#[derive(Debug, Clone, Default, Component)]
struct Acceleration(Vec2);

#[derive(Debug, Clone, Default, Component)]
struct Mass(f32);

#[derive(Debug, Clone, Default, Component)]
struct Radius(f32);

#[derive(Clone, Bundle, Default)]
struct Particle {
    position: Position,
    velocity: Velocity,
    acceleration: Acceleration,
    radius: Radius,
    mass: Mass,
    sprite: MaterialMesh2dBundle<ColorMaterial>,
}

fn rand_vec2_in_rect(rng: &mut impl Rng, rect: Rect) -> Vec2 {
    Vec2::new(
        rng.gen_range(rect.min.x..=rect.max.x),
        rng.gen_range(rect.min.y..=rect.max.y),
    )
}

impl Particle {
    fn new_rand(
        position_rect: Rect,
        velocity_rect: Rect,
        material: Handle<ColorMaterial>,
    ) -> Particle {
        let mut rng = thread_rng();
        Particle {
            velocity: Velocity(rand_vec2_in_rect(&mut rng, velocity_rect)),
            position: Position(rand_vec2_in_rect(&mut rng, position_rect)),
            mass: Mass(1.),
            sprite: MaterialMesh2dBundle {
                material,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Move sprites to entities' position. Assumes the window is [-1.,1.] in both x and y
fn sync_position_and_sprite(mut query: Query<(&mut Transform, &Position), Changed<Position>>) {
    query.par_iter_mut().for_each(|(mut transform, position)| {
        transform.translation = Vec3::new(position.0.x, position.0.y, 0.)
    });
}
/// Sync sprite with size
fn sync_radius_and_sprite(
    mut meshes: ResMut<Assets<Mesh>>,
    mut particles: Query<(&mut Mesh2dHandle, &Radius), Changed<Radius>>,
) {
    particles.iter_mut().for_each(|(mut sprite, radius)| {
        *sprite = meshes.add(shape::Circle::new(radius.0).into()).into();
    })
}
/// Change position using velocity.
fn apply_velocity(mut query: Query<(&mut Position, &Velocity)>) {
    query
        .par_iter_mut()
        .for_each(|(mut position, velocity)| position.0 += velocity.0);
}
/// Change velocity using acceleration then reset acceleration.
fn apply_acceleration(mut query: Query<(&mut Velocity, &mut Acceleration)>) {
    query
        .par_iter_mut()
        .for_each(|(mut velocity, mut acceleration)| {
            velocity.0 += acceleration.0;
            acceleration.0 = Vec2::ZERO;
        });
}
/// Apply gravitational attraction to acceleration
fn apply_gravity(
    mut accelerators: Query<(Entity, &Position, &mut Acceleration), With<Mass>>,
    attractors: Query<(Entity, &Position, &Mass)>,
) {
    /// Acceleration due to gravity towards other
    fn attraction_to(this: &Position, other: (&Position, &Mass)) -> Acceleration {
        let dp = other.0 .0 - this.0;
        let dp_len = dp.length();
        // Acceleration(dp / dp_len * G * other.1 .0 / dp_len.powi(2))
        // perf: simplification of above
        Acceleration(dp * G * other.1 .0 / dp_len.powi(3)) // self mass cancels when calculating acceleration
    }

    accelerators
        .par_iter_mut()
        .for_each(|(outer_entity, outer_position, mut accel)| {
            attractors
                .iter()
                .filter(|(inner_entity, _, _)| inner_entity != &outer_entity)
                .for_each(|(_, position, mass)| {
                    accel.0 += attraction_to(outer_position, (position, mass)).0
                })
        });
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
) {
    let mut to_despawn = HashSet::new();
    // Without grouping all sources at once this won't be deterministic in parallel because floats don't commute
    merge_target.iter().for_each(|target| {
        merge_source
            .iter()
            .filter(|source| {
                source.0 != target.0
                    // source in target
                    && (target.1.0.distance(source.1.0)) < (target.2.0)
            }) // keep entities that are not the same and contains in the target
            .for_each(|source| {
                // source smaller than target
                let target_mass = shared_query.component::<Mass>(target.0).0;
                let source_mass = shared_query.component::<Mass>(source.0).0;
                // only despawn smaller
                if target_mass > source_mass
                // or first encountered of same mass
                    || (target_mass == source_mass && !to_despawn.contains(&target.0))
                {
                    let mass_final = source_mass + target_mass;
                    // Merge velocity with momentum conservation
                    if let Ok(source_velocity) =
                        shared_query.get_component::<Velocity>(source.0).cloned()
                    {
                        if let Ok(mut target_velocity) =
                            shared_query.get_component_mut::<Velocity>(target.0)
                        {
                            target_velocity.0 = (source_mass * source_velocity.0
                                + target_mass * target_velocity.0)
                                / (mass_final);
                        }
                    }

                    // Combine mass
                    shared_query.component_mut::<Mass>(target.0).0 = mass_final;
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

    // Larger center attractor
    commands.spawn(Particle {
        mass: Mass(100.),
        ..Default::default()
    });
    commands.spawn(Particle {
        position: Position(Vec2::new(0.5, 0.6)),
        velocity: Velocity(Vec2::new(0., -0.02)),
        mass: Mass(500.),
        ..Default::default()
    });
    commands.spawn(Particle {
        position: Position(Vec2::new(0.5, -0.6)),
        velocity: Velocity(Vec2::new(0., 0.01)),
        mass: Mass(1000.),
        ..Default::default()
    });
}

#[derive(Default, Debug, Clone, Event)]
struct Energy(f32);

#[derive(Debug, Default, Clone, Resource)]
struct KineticGraphPoints {
    points: Vec<Energy>,
}
#[derive(Debug, Default, Clone, Resource)]
struct PotentialGraphPoints {
    points: Vec<Energy>,
}
fn kinetic_energy(query: Query<(&Velocity, &Mass)>, mut energy: ResMut<KineticGraphPoints>) {
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
    mut energy: ResMut<PotentialGraphPoints>,
) {
    // perf: found Mutex slightly faster than using std::sync::mpsc and equal to atomic_float::AtomicF32
    let acc = Mutex::new(0.);
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
    mut contexts: EguiContexts,
    kinetic_energy: Res<KineticGraphPoints>,
    potential_energy: Res<PotentialGraphPoints>,
) {
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
            Plot::new("kinetic")
                // .view_aspect(1.8)
                .auto_bounds_x()
                .auto_bounds_y()
                .legend(Legend::default())
                .show(&mut columns[0], |plot_ui| {
                    plot_ui.line(kinetic_line);
                });
            columns[1].label("Potential Energy");
            Plot::new("potential")
                .auto_bounds_x()
                .auto_bounds_y()
                .show(&mut columns[1], |plot_ui| plot_ui.line(potential_line));
        });
    });
}
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, apply_gravity)
        .add_systems(Update, apply_acceleration.after(apply_gravity))
        .add_systems(Update, apply_velocity.after(apply_acceleration))
        .add_systems(Update, merge_colliders.after(apply_acceleration)) // merge colliders ignores accelerations
        .add_systems(Update, sync_mass_and_radius.after(merge_colliders))
        .add_systems(Update, sync_radius_and_sprite.after(sync_mass_and_radius))
        .add_systems(Update, sync_position_and_sprite.after(apply_velocity))
        .add_systems(Update, kinetic_energy)
        .add_systems(Update, potential_energy)
        .insert_resource(KineticGraphPoints::default())
        .insert_resource(PotentialGraphPoints::default())
        .add_systems(Update, graph)
        .run();
}
