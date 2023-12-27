use bevy::{
    app::{App, Startup, Update},
    core_pipeline::core_2d::Camera2dBundle,
    ecs::{
        bundle::Bundle,
        component::Component,
        entity::Entity,
        query::With,
        system::{Commands, Query, Res, ResMut, Resource},
    },
    math::{Quat, Rect, Vec2, Vec3},
    sprite::{Sprite, SpriteBundle},
    transform::components::Transform,
    window::Window,
    DefaultPlugins,
};
use rand::{rngs, thread_rng, Rng};

#[derive(Debug, Clone, Default, Component)]
struct Position(Vec2);

#[derive(Debug, Clone, Default, Component)]
struct Velocity(Vec2);

#[derive(Debug, Clone, Default, Component)]
struct Acceleration(Vec2);

/// Gravitational constant
const G: f32 = 6.67430e-11;
// const G: f32 = 6.6e-21;

#[derive(Debug, Clone, Default, Component)]
struct Mass(f32);

#[derive(Clone, Bundle, Default)]
struct Particle {
    position: Position,
    velocity: Velocity,
    acceleration: Acceleration,
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
            position.0.x * window.width(),
            position.0.y * window.height(),
            0.,
        )
    });
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
/// Set the size of the particle sprites to their mass
fn sprite_proportional_to_mass(mut particles: Query<(&mut Sprite, &Mass)>) {
    particles.par_iter_mut().for_each(|(mut sprite, mass)| {
        sprite.custom_size = Some(Vec2 {
            x: mass.0.sqrt(),
            y: mass.0.sqrt(),
        })
    });
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn(Camera2dBundle::default());

    // Randomized initial particles
    commands.spawn_batch(
        std::iter::repeat_with(|| {
            Particle::new_rand(
                Rect::new(-0.1, -0.1, 0.1, 0.1),
                Rect::new(-0.001, -0.001, 0.002, 0.002),
            )
        })
        .take(10000),
    );

    commands
        .spawn(Particle {
            mass: Mass(1000.),
            ..Default::default()
        })
        .remove::<(Acceleration, Velocity)>();
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, apply_gravity)
        .add_systems(Update, apply_acceleration)
        .add_systems(Update, apply_velocity)
        .add_systems(Update, sync_position_and_sprite)
        .add_systems(Update, sprite_proportional_to_mass)
        .run();
}
