// Vertex shader
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>
};

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
fn density_at_point(samplePoint: vec2<f32>) -> f32 {
    var density = 0.0;
    let radius = 0.05;
    for (var i: u32 = 0; i < arrayLength(&particles); i++) {
        let d0 = abs(particles[i].position[0] - samplePoint[0]);
        let d1 = abs(particles[i].position[1] - samplePoint[1]);
        if (d0 < radius && d1 < radius) {
            let di = distance(samplePoint, particles[i].position);
            if (di < radius) {
                density += pow(min(radius-di, 1.0)*25.0, 3.0);
            }
        }
    }
    return density - 0.25;
}

//vertex shader: this one says where to draw things. Or something.
@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = vec3<f32>(model.color[0], model.color[1], model.color[2]);
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader: this one draws pixels?

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let position = vec2<f32>(in.clip_position[0]/600, in.clip_position[1]/600);
    let stepsize = 0.005;
    let dens = density_at_point(position);
    //let deltaY = dens - density_at_point(position + vec2<f32>(0, stepsize));
    if (dens > 0.0) {
        return vec4<f32>(0.0, 0.05, dens, 1.0);
    } else {
        return vec4<f32>(-dens, 0.0, 0.0, 1.0);
    }
}
