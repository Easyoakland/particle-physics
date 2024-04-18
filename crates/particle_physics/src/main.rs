use bevy::{
    app::App,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    DefaultPlugins,
};
use particle_physics::ParticlePhysicsPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ParticlePhysicsPlugin {
            graph: true,
            merge: true,
            substeps: particle_physics::Substeps(10),
        })
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .run();
}
