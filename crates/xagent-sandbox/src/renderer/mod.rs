pub mod camera;
pub mod font;
pub mod hud;

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::world::Mesh;

/// In-flight frame state returned by `render_frame` so the caller can
/// append additional render passes (e.g. egui) before submitting.
pub struct FrameContext {
    pub encoder: wgpu::CommandEncoder,
    pub surface_output: wgpu::SurfaceTexture,
    pub view: wgpu::TextureView,
}

use font::{TextItem, TextRenderer};
use hud::HudBar;

// ── Instanced rendering types ──────────────────────────────────────────

/// Per-instance data for GPU instancing: position, color, and scale.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub scale: f32,
    pub _pad: f32,
}

/// Per-vertex data for the instanced unit cube: local position + face shade.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstancedVertex {
    pub position: [f32; 3],
    pub shade: f32,
}

pub const INSTANCED_SHADER_SRC: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) shade: f32,
};
struct InstanceInput {
    @location(2) inst_position: vec3<f32>,
    @location(3) inst_color: vec3<f32>,
    @location(4) inst_scale: f32,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = vertex.position * instance.inst_scale + instance.inst_position;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = instance.inst_color * vertex.shade;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

pub const SHADER_SRC: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

/// Shader for HUD overlay bars — pass-through NDC positions, no uniforms.
pub const HUD_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 0.85);
}
"#;

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub render_pipeline: wgpu::RenderPipeline,
    pub depth_texture_view: wgpu::TextureView,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    /// Pipeline for the HUD overlay bars (no depth, alpha-blended).
    pub hud_pipeline: wgpu::RenderPipeline,
    /// Bitmap font text renderer.
    pub text_renderer: TextRenderer,
    // Instanced agent rendering
    pub instance_pipeline: wgpu::RenderPipeline,
    pub unit_cube_vb: wgpu::Buffer,
    pub unit_cube_ib: wgpu::Buffer,
    pub unit_cube_num_indices: u32,
    // Persistent HUD/text buffers (avoid per-frame GPU allocation)
    pub hud_vb: wgpu::Buffer,
    pub hud_ib: wgpu::Buffer,
    pub hud_num_indices: u32,
    pub text_vb: wgpu::Buffer,
    pub text_ib: wgpu::Buffer,
    pub text_num_indices: u32,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find a suitable GPU adapter");

        log::info!("Using adapter: {:?}", adapter.get_info());

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xagent-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Immediate)
        {
            wgpu::PresentMode::Immediate
        } else if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::AutoVsync
        };

        log::info!(
            "Present mode: {:?} (available: {:?})",
            present_mode,
            surface_caps.present_modes
        );

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_buffer"),
            contents: bytemuck::cast_slice(&glam::Mat4::IDENTITY.to_cols_array()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
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
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let depth_texture_view = Self::create_depth_texture(&device, width, height);

        // ── HUD overlay pipeline ───────────────────────────────────
        let hud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud_shader"),
            source: wgpu::ShaderSource::Wgsl(HUD_SHADER_SRC.into()),
        });

        let hud_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud_pipeline"),
            layout: Some(&hud_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &hud_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
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
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // no culling for UI quads
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        // ── Text renderer ──────────────────────────────────────────
        let text_renderer = TextRenderer::new(&device, &queue, surface_format);

        // ── Instanced agent pipeline ──────────────────────────────
        let instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(INSTANCED_SHADER_SRC.into()),
        });

        let instanced_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("instanced_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let instance_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instance_pipeline"),
            layout: Some(&instanced_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &instanced_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // Slot 0: per-vertex (InstancedVertex)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<InstancedVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                    // Slot 1: per-instance (InstanceData)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x3, // inst_position
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x3, // inst_color
                            },
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32, // inst_scale
                            },
                        ],
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &instanced_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        // ── Unit cube mesh (uploaded once) ─────────────────────────
        let (unit_cube_verts, unit_cube_idxs) = generate_unit_cube();
        let unit_cube_num_indices = unit_cube_idxs.len() as u32;

        let unit_cube_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unit_cube_vb"),
            contents: bytemuck::cast_slice(&unit_cube_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let unit_cube_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unit_cube_ib"),
            contents: bytemuck::cast_slice(&unit_cube_idxs),
            usage: wgpu::BufferUsages::INDEX,
        });

        // ── Persistent HUD buffers ─────────────────────────────────
        let hud_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud_vertex_buffer"),
            size: (256 * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hud_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud_index_buffer"),
            size: (512 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Persistent text buffers ────────────────────────────────
        let text_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text_vertex_buffer"),
            size: (1024 * std::mem::size_of::<font::TextVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let text_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text_index_buffer"),
            size: (2048 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            depth_texture_view,
            uniform_buffer,
            uniform_bind_group,
            hud_pipeline,
            text_renderer,
            instance_pipeline,
            unit_cube_vb,
            unit_cube_ib,
            unit_cube_num_indices,
            hud_vb,
            hud_ib,
            hud_num_indices: 0,
            text_vb,
            text_ib,
            text_num_indices: 0,
        }
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture_view = Self::create_depth_texture(&self.device, width, height);
        }
    }

    pub fn render(
        &mut self,
        meshes: &[GpuMesh],
        view_proj: &glam::Mat4,
    ) -> Result<(), wgpu::SurfaceError> {
        let refs: Vec<&GpuMesh> = meshes.iter().collect();
        let ctx = self.render_frame(&refs, view_proj, None, 0)?;
        self.finish_frame(ctx);
        Ok(())
    }

    pub fn render_refs(
        &mut self,
        meshes: &[&GpuMesh],
        view_proj: &glam::Mat4,
    ) -> Result<(), wgpu::SurfaceError> {
        let ctx = self.render_frame(meshes, view_proj, None, 0)?;
        self.finish_frame(ctx);
        Ok(())
    }

    /// Acquire the surface texture and create a command encoder.
    /// Use this when you want to manually orchestrate render passes (3D offscreen → egui surface).
    pub fn begin_frame(&mut self) -> Result<FrameContext, wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            });
        Ok(FrameContext {
            encoder,
            surface_output: output,
            view,
        })
    }

    /// Render the 3D scene + agents + HUD + text to an offscreen texture pair.
    /// The encoder is provided externally so that egui can append its pass afterwards.
    pub fn render_3d_offscreen(
        &mut self,
        meshes: &[&GpuMesh],
        view_proj: &glam::Mat4,
        agent_instance_buffer: Option<&wgpu::Buffer>,
        agent_instance_count: u32,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&view_proj.to_cols_array()),
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("offscreen_3d_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.12,
                            g: 0.12,
                            b: 0.14,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 3D scene
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            for mesh in meshes {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }

            // Instanced agents
            if let Some(inst_buf) = agent_instance_buffer {
                if agent_instance_count > 0 {
                    let inst_stride = std::mem::size_of::<InstanceData>() as u64;
                    pass.set_pipeline(&self.instance_pipeline);
                    pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    pass.set_vertex_buffer(0, self.unit_cube_vb.slice(..));
                    pass.set_vertex_buffer(
                        1,
                        inst_buf.slice(..agent_instance_count as u64 * inst_stride),
                    );
                    pass.set_index_buffer(self.unit_cube_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.unit_cube_num_indices, 0, 0..agent_instance_count);
                }
            }

            // HUD overlay
            if self.hud_num_indices > 0 {
                pass.set_pipeline(&self.hud_pipeline);
                pass.set_vertex_buffer(0, self.hud_vb.slice(..));
                pass.set_index_buffer(self.hud_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.hud_num_indices, 0, 0..1);
            }

            // Text overlay
            if self.text_num_indices > 0 {
                pass.set_pipeline(&self.text_renderer.pipeline);
                pass.set_bind_group(0, &self.text_renderer.bind_group, &[]);
                pass.set_vertex_buffer(0, self.text_vb.slice(..));
                pass.set_index_buffer(self.text_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.text_num_indices, 0, 0..1);
            }
        }
    }

    /// Update HUD overlay geometry. Call only when HUD content changes.
    pub fn update_hud(&mut self, bars: &[HudBar], panels: &[(f32, f32, f32, f32)]) {
        let mut all_verts: Vec<Vertex> = Vec::new();
        let mut all_idxs: Vec<u32> = Vec::new();

        for &(px, py, pw, ph) in panels {
            let (pv, pi) = hud::build_panel_vertices(px, py, pw, ph);
            let base = all_verts.len() as u32;
            all_verts.extend_from_slice(&pv);
            all_idxs.extend(pi.iter().map(|i| i + base));
        }

        if !bars.is_empty() {
            let (bv, bi) = hud::build_hud_vertices(bars);
            let base = all_verts.len() as u32;
            all_verts.extend_from_slice(&bv);
            all_idxs.extend(bi.iter().map(|i| i + base));
        }

        if !all_verts.is_empty() {
            self.queue
                .write_buffer(&self.hud_vb, 0, bytemuck::cast_slice(&all_verts));
            self.queue
                .write_buffer(&self.hud_ib, 0, bytemuck::cast_slice(&all_idxs));
        }
        self.hud_num_indices = all_idxs.len() as u32;
    }

    /// Update text overlay geometry. Call only when text content changes.
    pub fn update_text(&mut self, items: &[TextItem]) {
        if items.is_empty() {
            self.text_num_indices = 0;
            return;
        }
        let (text_verts, text_idxs) = self.text_renderer.build_texts(items);
        if !text_verts.is_empty() {
            self.queue
                .write_buffer(&self.text_vb, 0, bytemuck::cast_slice(&text_verts));
            self.queue
                .write_buffer(&self.text_ib, 0, bytemuck::cast_slice(&text_idxs));
        }
        self.text_num_indices = text_idxs.len() as u32;
    }

    /// Render one frame using a single render pass. Draws 3D meshes, instanced
    /// agents, then HUD and text overlays using previously uploaded geometry.
    pub fn render_frame(
        &mut self,
        meshes: &[&GpuMesh],
        view_proj: &glam::Mat4,
        agent_instance_buffer: Option<&wgpu::Buffer>,
        agent_instance_count: u32,
    ) -> Result<FrameContext, wgpu::SurfaceError> {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&view_proj.to_cols_array()),
        );

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        // Single render pass for everything: 3D scene → agents → HUD → text
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.12,
                            g: 0.12,
                            b: 0.14,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard, // tile-based GPU: skip writeback
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // ── 3D scene ──────────────────────────────────────
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            for mesh in meshes {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }

            // ── Instanced agents ──────────────────────────────
            if let Some(inst_buf) = agent_instance_buffer {
                if agent_instance_count > 0 {
                    let inst_stride = std::mem::size_of::<InstanceData>() as u64;
                    pass.set_pipeline(&self.instance_pipeline);
                    pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    pass.set_vertex_buffer(0, self.unit_cube_vb.slice(..));
                    pass.set_vertex_buffer(
                        1,
                        inst_buf.slice(..agent_instance_count as u64 * inst_stride),
                    );
                    pass.set_index_buffer(self.unit_cube_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.unit_cube_num_indices, 0, 0..agent_instance_count);
                }
            }

            // ── HUD overlay (drawn on top, depth always passes) ──
            if self.hud_num_indices > 0 {
                pass.set_pipeline(&self.hud_pipeline);
                pass.set_vertex_buffer(0, self.hud_vb.slice(..));
                pass.set_index_buffer(self.hud_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.hud_num_indices, 0, 0..1);
            }

            // ── Text overlay (drawn on top, depth always passes) ──
            if self.text_num_indices > 0 {
                pass.set_pipeline(&self.text_renderer.pipeline);
                pass.set_bind_group(0, &self.text_renderer.bind_group, &[]);
                pass.set_vertex_buffer(0, self.text_vb.slice(..));
                pass.set_index_buffer(self.text_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.text_num_indices, 0, 0..1);
            }
        }

        Ok(FrameContext {
            encoder,
            surface_output: output,
            view,
        })
    }

    /// Submit the command buffer and present the surface.
    /// Call this after appending any additional render passes (e.g. egui) to `ctx.encoder`.
    pub fn finish_frame(&self, ctx: FrameContext) {
        self.queue.submit(std::iter::once(ctx.encoder.finish()));
        ctx.surface_output.present();
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl GpuMesh {
    pub fn from_mesh(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: mesh.indices.len() as u32,
        }
    }

    /// Create a GPU mesh backed by writable buffers suitable for `queue.write_buffer()`.
    pub fn new_dynamic(device: &wgpu::Device, max_vertices: u64, max_indices: u64) -> Self {
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic_vertex_buffer"),
            size: max_vertices * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic_index_buffer"),
            size: max_indices * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            vertex_buffer,
            index_buffer,
            num_indices: 0,
        }
    }

    /// Update this dynamic mesh's GPU data via queue.write_buffer (no allocation).
    pub fn update_from_mesh(&mut self, queue: &wgpu::Queue, mesh: &Mesh) {
        if !mesh.vertices.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&mesh.vertices));
        }
        if !mesh.indices.is_empty() {
            queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&mesh.indices));
        }
        self.num_indices = mesh.indices.len() as u32;
    }
}

/// Generate a centered unit cube (half_size=1.0) with per-face shading.
/// Returns (vertices, indices) — 24 InstancedVertex + 36 u32 indices.
fn generate_unit_cube() -> (Vec<InstancedVertex>, Vec<u32>) {
    let h: f32 = 1.0;
    #[rustfmt::skip]
    let positions: [[f32; 3]; 8] = [
        [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
        [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],
    ];

    let shades: [f32; 6] = [1.0, 0.9, 0.85, 0.7, 0.8, 0.75];
    #[rustfmt::skip]
    let face_verts: [(usize, usize, usize, usize); 6] = [
        (0, 1, 2, 3), (5, 4, 7, 6), (3, 2, 6, 7),
        (4, 5, 1, 0), (4, 0, 3, 7), (1, 5, 6, 2),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (fi, &(a, b, c, d)) in face_verts.iter().enumerate() {
        let base = (fi * 4) as u32;
        let shade = shades[fi];

        vertices.push(InstancedVertex {
            position: positions[a],
            shade,
        });
        vertices.push(InstancedVertex {
            position: positions[b],
            shade,
        });
        vertices.push(InstancedVertex {
            position: positions[c],
            shade,
        });
        vertices.push(InstancedVertex {
            position: positions[d],
            shade,
        });

        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (vertices, indices)
}
