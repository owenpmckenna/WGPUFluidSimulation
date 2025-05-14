struct Particle {
    @location(0) position: vec2<f32>,
    @location(1) velocity: vec2<f32>
};
struct OuterActionState {
    @location(0) mouse_x: f32,
    @location(1) mouse_y: f32
}
@group(0)
@binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0)
@binding(1)
var<storage, read_write> outer_action_state: array<OuterActionState>;


fn smooth_(radius: f32, dst: f32) -> f32 {
    //var val = max(0.0, radius - dst);
    return pow(max(0.0, radius - dst), 3.0);
}
fn density_at_point_optimised(samplePoint: vec2<f32>, forid: u32, stepsize: f32) -> vec3<f32> {
    var home_density = 0.0;
    var diffy_density = vec2<f32>(0.0, 0.0);
    let testing_radius = 0.02 + stepsize;
    let radius = 0.02;
    for (var i: u32 = 0; i < arrayLength(&particles); i++) {
        if (i == forid) {
            continue;
        }
        let d = particles[i].position.xy - samplePoint.xy;
        var di = 0.0;
        if (d.x < testing_radius && d.y < testing_radius) {
            if (d.x < radius && d.y < radius) {
                di = distance(samplePoint, particles[i].position);
                if (di < radius) {
                    home_density += pow(min(radius-di, 1.0)*25.0, 3.0);
                }
            }

            if (d.x < radius + stepsize && d.y < radius) {
                di = distance(samplePoint + vec2<f32>(stepsize, 0.0), particles[i].position);
                if (di < radius) {
                    diffy_density.x += pow(min(radius-di, 1.0)*25.0, 3.0);
                }
            }

            if (d.x < radius && d.y < radius + stepsize) {
                di = distance(samplePoint + vec2<f32>(0.0, stepsize), particles[i].position);
                if (di < radius) {
                    diffy_density.y += pow(min(radius-di, 1.0)*25.0, 3.0);
                }
            }
        }
    }
    var out_density = home_density - diffy_density;
    return vec3<f32>(out_density.xy, home_density) + mouse_density(samplePoint);
}
fn mouse_density(samplePoint: vec2<f32>) -> f32 {
    let radius = 0.2;
    let d0 = abs(outer_action_state[0].mouse_x/600 - samplePoint[0]);
    let d1 = abs(outer_action_state[0].mouse_y/600 - samplePoint[1]);
    var density = 0.0;
    if (d0 < radius && d1 < radius) {
        let di = distance(samplePoint, vec2<f32>(outer_action_state[0].mouse_x/600, outer_action_state[0].mouse_y/600));
        if (di < radius) {
            density += pow(min(radius-di, 1.0)*35.0, 3.0);
        }
    }
    return density;
}

@compute @workgroup_size(16, 4, 4)
fn cs_main(
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let workgroup_index =
     workgroup_id.x +
     workgroup_id.y * num_workgroups.x +
     workgroup_id.z * num_workgroups.x * num_workgroups.y;
  let global_invocation_index =
     workgroup_index * 256 +
     local_invocation_index;
    let idx = global_invocation_index;//id.x; // Global ID for this work item

    // Check that the index is within bounds
    if (idx < arrayLength(&particles)) {
        // Update particle position based on velocity

        //let direction_effect = vec2<f32> =
        particles[idx].velocity[1] += 0.5;
        //particles[idx].velocity[1] += 0.03;//positive y is down now.
        //how tf is this going to work???
        let stepsize = 0.005;
        //let dens = density_at_point(particles[idx].position, idx);
        let dens_data = density_at_point_optimised(particles[idx].position, idx, stepsize);
        particles[idx].velocity += dens_data.xy;
        let dens = dens_data.z;
        //if density is less to the + of us: deltaX will be + and we'll go in that direction
        //let deltaX = dens - density_at_point(particles[idx].position + vec2<f32>(stepsize, 0), idx);
        //let deltaY = dens - density_at_point(particles[idx].position + vec2<f32>(0, stepsize), idx);
        //particles[idx].velocity[0] += deltaX;
        //particles[idx].velocity[1] += deltaY;
        //let deltaX0 = dens - density_at_point(particles[idx].position + vec2<f32>(-stepsize, 0), idx);
        //let deltaY0 = dens - density_at_point(particles[idx].position + vec2<f32>(0, -stepsize), idx);
        //particles[idx].velocity[0] -= deltaX0;
        //particles[idx].velocity[1] -= deltaY0;
        if (dens < 0.0) {
            particles[idx].velocity *= 0.999;
        }
        let particle_size = 0.01;
        let amt = 0.005;
        let amtmul = -0.5;
        if (particles[idx].position[0] > 1.0 - particle_size) {
            particles[idx].velocity[0] *= amtmul;
            particles[idx].position[0] -= amt;
        }
        if (particles[idx].position[1] > 1.0 - particle_size) {
            particles[idx].velocity[1] *= amtmul;
            particles[idx].position[1] -= amt;
        }
        if (particles[idx].position[0] < 0.0 + particle_size) {
            particles[idx].velocity[0] *= amtmul;
            particles[idx].position[0] += amt;
        }
        if (particles[idx].position[1] < 0.0 + particle_size) {
            particles[idx].velocity[1] *= amtmul;
            particles[idx].position[1] += amt;
        }
        particles[idx].position += particles[idx].velocity * 0.0005;
    }
}