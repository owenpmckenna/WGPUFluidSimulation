use std::mem::size_of_val;
use std::ops::ControlFlow;
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use crossbeam_channel::{unbounded, Receiver, Sender};
use pollster::block_on;
use wgpu::{BindGroup, BindGroupLayoutDescriptor, BindingType, BufferBindingType, BufferSize, ComputePipeline, Device, Label, MemoryHints, PipelineLayout, PipelineLayoutDescriptor, PresentMode, Queue, ShaderStages};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{WindowBuilder, Window},
};
use wgpu::util::DeviceExt;
use winit::dpi::{PhysicalSize, Size};
use winit::event_loop::{EventLoopBuilder, EventLoopWindowTarget};
use winit::platform::windows::EventLoopBuilderExtWindows;

const sleeptime: u64 = 1;
const numparticles: u32 = 4000;
const USE_TESTING_SHADER: bool = false;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
}
impl Particle {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                }
            ]
        }
    }
}
struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    vertex_particle_buffer: wgpu::Buffer,
    bind_group: BindGroup,
    num_particles: u32,
    num_indices: u32,
    indices: Vec<u16>,
    vertices: Vec<Vertex>,
    compute_state: ComputeState,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: &'a Window,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OuterActionState {
    mouse_x: f32,
    mouse_y: f32
}
struct ComputeState {
    storage_buffer: wgpu::Buffer,
    outer_action_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_storage: wgpu::BindGroup,
    outer_action_state: OuterActionState
}
fn setup_compute(device: &Device, particles: &Vec<Particle>) -> ComputeState {
    let cs_module = device.create_shader_module(wgpu::include_wgsl!("compute_shader.wgsl"));
    let size = size_of_val(particles) as wgpu::BufferAddress;

    let outer_action_state = OuterActionState { mouse_x: 500.0, mouse_y: 560.0 };

    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(particles),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let outer_action_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Outer Action Buffer"),
        contents: bytemuck::cast_slice(&mut [outer_action_state]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let bind_group_layout_entry0 = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(BufferSize::try_from(storage_buffer.size()).unwrap()),
        },
        count: None,
    };
    let bind_group_layout_entry1 = wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(BufferSize::try_from(outer_action_buffer.size()).unwrap()),
        },
        count: None,
    };
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[bind_group_layout_entry0, bind_group_layout_entry1],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "cs_main",
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: outer_action_buffer.as_entire_binding(),
            }
        ],
    });
    ComputeState {storage_buffer, compute_pipeline, bind_group_storage, outer_action_state, outer_action_buffer}
}
impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window) -> State<'a> {
        let vertices = vec!(
            Vertex { position: [-1.0, -1.0, 0.0], color: [0.5, 0.0, 0.5] },
            Vertex { position: [-1.0, 1.0, 0.0], color: [0.0, 0.5, 0.5] },
            Vertex { position: [1.0, -1.0, 0.0], color: [0.0, 0.5, 0.5] },
            Vertex { position: [1.0, 1.0, 0.0], color: [0.5, 0.0, 0.5] },
        );
        let indices = vec!(
            0, 1, 2,
            1, 2, 3,
            0, 0, 0,
        );

        let num_particles = numparticles;
        let mut particles = Vec::new();
        for _ in 0..num_particles {
            particles.push(Particle {
                position: [0.5 + rand::random::<f32>()/5.0, 0.5 + rand::random::<f32>()/5.0],
                velocity: //[0.0, rand::random::<f32>()/2.0]
                [rand::random::<f32>()*2.0 - 1.0, rand::random::<f32>()*2.0 - 1.0]
            });
        }

        let num_indices = indices.len() as u32;

        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch="wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch="wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: wgpu::Limits::default(),
                label: None,
                memory_hints: MemoryHints::Performance,
            },
            None, // Trace path
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(
                match USE_TESTING_SHADER {
                    true => include_str!("shader_test.wgsl"),
                    false => include_str!("shader.wgsl")
                }
                    .into()),
        });

        let vertex_particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&particles),
            //usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: Some(BufferSize::try_from(vertex_particle_buffer.size()).unwrap()),
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_particle_buffer.as_entire_binding(),
            }],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // 1.
                buffers: &[
                    Vertex::desc()
                ], // 2.
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: None,//Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1, // 2.
                mask: !0, // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
            cache: None, // 6.
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        let data = bytemuck::cast_slice(indices.as_slice());
        println!("v{}, d{}", indices.len(), data.len());
        println!("size: w:{} h:{}", size.width, size.height);
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: data,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        let compute_state = setup_compute(&device, &particles);

        Self { window, surface, device, queue, config, size, render_pipeline, compute_state, vertex_buffer, index_buffer, vertex_particle_buffer, num_particles, bind_group, num_indices, vertices, indices }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        let mut start = Instant::now();
        self.queue.write_buffer(&self.compute_state.outer_action_buffer, 0, bytemuck::cast_slice([self.compute_state.outer_action_state].as_slice()));
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_state.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_state.bind_group_storage, &[]);
            cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch_workgroups(self.num_particles, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        println!("compute duration: {}", start.elapsed().as_millis());
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let start = Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.compute_state.storage_buffer, 0, &self.vertex_particle_buffer, 0, self.compute_state.storage_buffer.size());
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.5,
                            b: 0.5,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.bind_group, &[]);//um... will this work???
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        println!("draw duration: {}", start.elapsed().as_millis());

        Ok(())
    }
}
fn handle_window_event(event: &WindowEvent, control_flow: &EventLoopWindowTarget<()>, state: &mut State) {
    match event {
        WindowEvent::CloseRequested
        | WindowEvent::KeyboardInput {
            event:
            KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Escape),
                ..
            },
            ..
        } => control_flow.exit(),
        WindowEvent::Resized(physical_size) => {
            state.resize(*physical_size);
        },
        WindowEvent::RedrawRequested => {
            //if !surface_configured {
            //	return;
            //}
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if it's lost or outdated
                Err(
                    wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                ) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    log::error!("OutOfMemory");
                    control_flow.exit();
                }

                // This happens when the a frame takes too long to present
                Err(wgpu::SurfaceError::Timeout) => {
                    log::warn!("Surface timeout")
                }
            }

            // This tells winit that we want another frame after this one
            sleep(Duration::from_millis(sleeptime));
            //println!("redraw!");
            state.window().request_redraw();
        },
        WindowEvent::CursorMoved { position, .. } => {
            state.compute_state.outer_action_state.mouse_x = position.x as f32;
            state.compute_state.outer_action_state.mouse_y = position.y as f32;
        },
        WindowEvent::CursorLeft {..} => {
            state.compute_state.outer_action_state.mouse_x = 0.0;
            state.compute_state.outer_action_state.mouse_y = 0.0;
        },
        _ => {}
    }
}
fn go(event_loop: EventLoop<()>, mut state: State) {
    let d = event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => if !state.input(event) {
                handle_window_event(event, control_flow, &mut state);
            },
            _ => {}
        }
    });
}
pub fn main() {
    env_logger::init();
    let event_loop = EventLoopBuilder::new().with_any_thread(true).build().unwrap();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(600, 600))
        .build(&event_loop)
        .unwrap();
    let state = block_on(State::new(&window));
    go(event_loop, state);
    println!("prep!");
}

