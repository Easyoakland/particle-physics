use bevy::{
    asset::Handle,
    ecs::{bundle::Bundle, component::Component},
    math::{Rect, Vec2},
    sprite::{ColorMaterial, MaterialMesh2dBundle},
};

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct Position(pub Vec2);

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct Velocity(pub Vec2);

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct Acceleration(pub Vec2);

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct NewAcceleration(pub Vec2);

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct Mass(pub f32);

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct Radius(pub f32);

#[derive(Clone, Bundle, Default)]
pub struct Particle {
    pub position: Position,
    pub velocity: Velocity,
    pub acceleration: Acceleration,
    pub new_acceleration: NewAcceleration,
    pub radius: Radius,
    pub mass: Mass,
    pub sprite: MaterialMesh2dBundle<ColorMaterial>,
}
impl Particle {
    pub fn new_rand(
        position_rect: Rect,
        velocity_rect: Rect,
        material: Handle<ColorMaterial>,
    ) -> Particle {
        fn rand_vec2_in_rect(rng: &mut impl rand::Rng, rect: Rect) -> Vec2 {
            Vec2::new(
                rng.gen_range(rect.min.x..=rect.max.x),
                rng.gen_range(rect.min.y..=rect.max.y),
            )
        }
        let mut rng = rand::thread_rng();
        Particle {
            velocity: Velocity(rand_vec2_in_rect(&mut rng, velocity_rect)),
            position: Position(rand_vec2_in_rect(&mut rng, position_rect)),
            mass: Mass(1.),
            sprite: bevy::sprite::MaterialMesh2dBundle {
                material,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}
