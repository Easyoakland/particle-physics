use bevy::{
    app::{App, PluginGroup},
    core::{TaskPoolOptions, TaskPoolPlugin, TaskPoolThreadAssignmentPolicy},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    DefaultPlugins,
};
use particle_physics::ParticlePhysicsPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions {
                min_total_threads: 1,
                max_total_threads: usize::MAX,
                io: TaskPoolThreadAssignmentPolicy {
                    min_threads: 1,
                    max_threads: usize::MAX,
                    percent: 0.,
                },
                async_compute: TaskPoolThreadAssignmentPolicy {
                    min_threads: 1,
                    max_threads: usize::MAX,
                    percent: 0.,
                },
                compute: TaskPoolThreadAssignmentPolicy {
                    min_threads: 1,
                    max_threads: usize::MAX,
                    percent: 1.,
                },
            },
        }))
        .add_plugins(ParticlePhysicsPlugin {
            graph: true,
            merge: true,
            substeps: particle_physics::Substeps(1),
        })
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .run();
}
