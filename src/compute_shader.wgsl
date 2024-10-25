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
fn smooth_(radius: f32, dst: f32) -> f32 {
    //var val = max(0.0, radius - dst);
    return pow(min(max(0.0, radius - dst), 1.0), 13.0);
}

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x; // Global ID for this work item

    // Check that the index is within bounds
    if (idx < arrayLength(&particles)) {
        // Update particle position based on velocity

        //let direction_effect = vec2<f32> =
        //particles[idx].velocity[1] -= 0.01;
        if (particles[idx].position[0] > 1.0 || particles[idx].position[0] < 0.0) {
            particles[idx].velocity[0] *= -1.0;
        }
        if (particles[idx].position[1] > 1.0 || particles[idx].position[1] < 0.0) {
            particles[idx].velocity[1] *= -1.0;
        }
        particles[idx].position += particles[idx].velocity * 0.020;
    }
}