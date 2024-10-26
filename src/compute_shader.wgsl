struct Particle {
    @location(0) position: vec2<f32>,
    @location(1) velocity: vec2<f32>
};
@binding(0)
@group(0)
var<storage, read_write> particles: array<Particle>;


fn dist(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
}
fn distance(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return dist(a[0], a[1], b[0], b[1]);
}
fn smooth_(radius: f32, dst: f32) -> f32 {
    //var val = max(0.0, radius - dst);
    return pow(max(0.0, radius - dst), 3.0);
}
fn density_at_point(samplePoint: vec2<f32>, forid: u32) -> f32 {
    var density = 0.0;
    let radius = 0.05;
    for (var i: u32 = 0; i < arrayLength(&particles); i++) {
        if (i == forid) {
            continue;
        }
        let d0 = abs(particles[i].position[0] - samplePoint[0]);
        let d1 = abs(particles[i].position[1] - samplePoint[1]);
        if (d0 < radius && d1 < radius) {
            let di = distance(samplePoint, particles[i].position);
            if (di < radius) {
                density += pow(min(radius-di, 1.0)*25.0, 3.0);
            }
        }
    }
    return (density - 0.25)/10;
}

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x; // Global ID for this work item

    // Check that the index is within bounds
    if (idx < arrayLength(&particles)) {
        // Update particle position based on velocity

        //let direction_effect = vec2<f32> =
        particles[idx].velocity[1] += 0.005;
        //particles[idx].velocity[1] += 0.03;//positive y is down now.
        //how tf is this going to work???
        let stepsize = 0.005;
        let dens = density_at_point(particles[idx].position, idx);
        //if density is less to the + of us: deltaX will be + and we'll go in that direction
        let deltaX = dens - density_at_point(particles[idx].position + vec2<f32>(stepsize, 0), idx);
        let deltaY = dens - density_at_point(particles[idx].position + vec2<f32>(0, stepsize), idx);
        particles[idx].velocity[0] += deltaX;
        particles[idx].velocity[1] += deltaY;
        let deltaX0 = dens - density_at_point(particles[idx].position + vec2<f32>(-stepsize, 0), idx);
        let deltaY0 = dens - density_at_point(particles[idx].position + vec2<f32>(0, -stepsize), idx);
        particles[idx].velocity[0] -= deltaX0;
        particles[idx].velocity[1] -= deltaY0;
        if (dens < 0.0) {
            particles[idx].velocity *= 0.90;
        }
        let particle_size = 0.01;
        let amt = 0.001;
        let amtmul = -0.8;
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
        particles[idx].position += particles[idx].velocity * 0.0025;
    }
}