use bevy::{
    app::{App, Startup, Update},
    core_pipeline::core_2d::Camera2dBundle,
    ecs::{
        bundle::Bundle,
        component::Component,
        entity::Entity,
        query::With,
        schedule::IntoSystemConfigs,
        system::{Commands, Query},
    },
    math::{Rect, Vec2, Vec3},
    sprite::{Sprite, SpriteBundle},
    transform::components::Transform,
    utils::hashbrown::HashSet,
    window::Window,
    DefaultPlugins,
};
use rand::{thread_rng, Rng};

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
    sprite: SpriteBundle,
}

fn rand_vec2_in_rect(rng: &mut impl Rng, rect: Rect) -> Vec2 {
    Vec2::new(
        rng.gen_range(rect.min.x..=rect.max.x),
        rng.gen_range(rect.min.y..=rect.max.y),
    )
}

impl Particle {
    fn new_rand(position_rect: Rect, velocity_rect: Rect) -> Particle {
        let mut rng = thread_rng();
        Particle {
            velocity: Velocity(rand_vec2_in_rect(&mut rng, velocity_rect)),
            position: Position(rand_vec2_in_rect(&mut rng, position_rect)),
            mass: Mass(1.),
            sprite: SpriteBundle {
                sprite: Sprite {
                    color: bevy::render::color::Color::Rgba {
                        red: 1.,
                        green: 0.,
                        blue: 0.,
                        alpha: 1.,
                    },
                    custom_size: Some(Vec2 { x: 1., y: 1. }),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Move sprites to entities' position. Assumes the window is [-1.,1.] in both x and y
fn sync_position_and_sprite(mut query: Query<(&mut Transform, &Position)>, window: Query<&Window>) {
    let window = window.single();
    query.par_iter_mut().for_each(|(mut transform, position)| {
        transform.translation = Vec3::new(
            position.0.x * window.width() / 2.,
            position.0.y * window.height() / 2.,
            0.,
        )
    });
}
/// Sync sprite with size
fn sync_radius_and_sprite(window: Query<&Window>, mut particles: Query<(&mut Sprite, &Radius)>) {
    let window = window.single();
    particles.par_iter_mut().for_each(|(mut sprite, radius)| {
        sprite.custom_size = Some(Vec2::new(
            radius.0 * window.width(),
            radius.0 * window.height(),
        ))
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
    /// Gravitational constant
    const G: f32 = 6.67430e-11;
    // const G: f32 = 6.6e-21;

    /// Acceleration due to gravity towards other
    fn attraction_to(this: &Position, other: (&Position, &Mass)) -> Acceleration {
        let dp = other.0 .0 - this.0;
        Acceleration(dp.normalize() * G * other.1 .0 / dp.length().powi(2)) // self mass cancels when calculating acceleration
    }
    /// Apply attraction to `other` onto `this`
    fn attract_to(this: (&Position, &mut Acceleration), other: (&Position, &Mass)) {
        this.1 .0 += attraction_to(this.0, other).0
    }

    accelerators
        .par_iter_mut()
        .for_each(|(outer_entity, outer_position, mut accel)| {
            attractors
                .iter()
                .filter(|(inner_entity, _, _)| inner_entity != &outer_entity)
                .for_each(|(_, position, mass)| {
                    attract_to((outer_position, &mut *accel), (position, mass))
                })
        });
}
/// Set the size of objects proportional to their mass
fn sync_mass_and_radius(mut particles: Query<(&mut Radius, &Mass)>) {
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
    // Without grouping all sources at once this won't be deterministic because floats don't commute
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

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn(Camera2dBundle::default());

    // Randomized initial particles
    commands.spawn_batch(
        std::iter::repeat_with(|| {
            Particle::new_rand(
                Rect::new(-0.4, -0.4, 0.4, 0.4),
                Rect::new(-0.001, -0.001, 0.001, 0.001),
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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, apply_gravity)
        .add_systems(Update, apply_acceleration)
        .add_systems(Update, apply_velocity)
        .add_systems(Update, sync_position_and_sprite)
        .add_systems(Update, merge_colliders.after(apply_acceleration)) // merge colliders ignores accelerations
        .add_systems(Update, sync_mass_and_radius)
        .add_systems(Update, sync_radius_and_sprite)
        .run();
}