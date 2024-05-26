// This shader is used for the gpu_readback example
// The actual work it does is not important for the example

@group(0) @binding(0) var<storage, read_write> accelerations: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> masses: array<f32>;
@group(0) @binding(3) var<storage, read_write> len: u32;

const G = 6.67430e-11;

/// Acceleration due to gravity towards other
fn attraction_to(this_pos: vec2<f32>, other_pos: vec2<f32>, other_mass: f32) -> vec2<f32> {
    let dp = other_pos - this_pos;
    let denom = pow(length(dp), 3f);
    if denom == 0f {
        return vec2f();
    } else {
        return dp * other_mass / denom;
    }
}

// The workgroup size is a sub-grid per workgroup for multiple invocations per grid-point cell
// in the overall dispatch grid.
// As indicated at <https://computergraphics.stackexchange.com/questions/13437/workgroup-size-performance-change>
// and <https://computergraphics.stackexchange.com/questions/12462/compute-shader-workgroups-execution-and-size>
// this is optimal when a multiple of architecure's expected size e.g. 32 for Nvidia (a warp) and 64 on AMD (a wavefront).
// So 64 would be optimal for either and 32 only for Nvidia while AMD would be idling half the threads at size of 32.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let this_idx = global_id.x;
    var new_accel = vec2f(0);

    if this_idx > len {
        return;
    }

    let this_pos = positions[this_idx];
    for (var idx = 0u; idx < len; idx++) {
        if idx == this_idx {
            continue;
        }
        // forall other
        new_accel += attraction_to(this_pos, positions[idx], masses[idx]);
    }

    // Wait for all new values to be calculated
    storageBarrier();

    // Write new values
    accelerations[this_idx] = new_accel * G;
}