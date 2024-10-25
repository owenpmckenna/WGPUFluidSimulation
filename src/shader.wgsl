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
fn smooth_(radius: f32, dst: f32) -> f32 {
    //var val = max(0.0, radius - dst);
    return pow(min(max(0.0, radius - dst), 1.0), 13.0);
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
    //return vec4<f32>(in.color, 1.0);
    //closest?
    for (var i: u32 = 0; i < arrayLength(&particles); i++) {
        let d0 = abs(particles[i].position[0] - in.clip_position[0]/600);
        let d1 = abs(particles[i].position[1] - in.clip_position[1]/600);
        if (d0 < 0.01 && d1 < 0.01) {
            let di = dist(particles[i].position[0], particles[i].position[1], in.clip_position[0]/600, in.clip_position[1]/600);
            if (di < 0.01) {
                return vec4<f32>(particles[i].position[0], particles[i].velocity, 1.0);
            }
        }
    }
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
}
