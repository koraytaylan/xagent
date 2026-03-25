//! Procedural bitmap font atlas and text quad generation.

use wgpu::util::DeviceExt;

/// Width of each glyph in the bitmap font atlas, in pixels.
pub const GLYPH_W: u32 = 8;
/// Height of each glyph in the bitmap font atlas, in pixels.
pub const GLYPH_H: u32 = 12;

const CHARSET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz%.: /-+#()";
const GLYPHS_PER_ROW: u32 = 16;

fn char_index(ch: char) -> Option<usize> {
    CHARSET.find(ch)
}

type GlyphBits = [u8; 12];

fn glyph_bits(ch: char) -> GlyphBits {
    match ch {
        '0' => [0x3C,0x66,0x6E,0x7E,0x76,0x66,0x66,0x66,0x66,0x3C,0x00,0x00],
        '1' => [0x18,0x38,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x00],
        '2' => [0x3C,0x66,0x06,0x06,0x0C,0x18,0x30,0x60,0x66,0x7E,0x00,0x00],
        '3' => [0x3C,0x66,0x06,0x06,0x1C,0x06,0x06,0x06,0x66,0x3C,0x00,0x00],
        '4' => [0x0C,0x1C,0x2C,0x4C,0x4C,0x7E,0x0C,0x0C,0x0C,0x0C,0x00,0x00],
        '5' => [0x7E,0x60,0x60,0x7C,0x06,0x06,0x06,0x06,0x66,0x3C,0x00,0x00],
        '6' => [0x3C,0x66,0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x3C,0x00,0x00],
        '7' => [0x7E,0x66,0x06,0x0C,0x0C,0x18,0x18,0x18,0x18,0x18,0x00,0x00],
        '8' => [0x3C,0x66,0x66,0x66,0x3C,0x66,0x66,0x66,0x66,0x3C,0x00,0x00],
        '9' => [0x3C,0x66,0x66,0x66,0x3E,0x06,0x06,0x06,0x66,0x3C,0x00,0x00],
        'A' => [0x18,0x3C,0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x66,0x00,0x00],
        'B' => [0x7C,0x66,0x66,0x66,0x7C,0x66,0x66,0x66,0x66,0x7C,0x00,0x00],
        'C' => [0x3C,0x66,0x60,0x60,0x60,0x60,0x60,0x60,0x66,0x3C,0x00,0x00],
        'D' => [0x78,0x6C,0x66,0x66,0x66,0x66,0x66,0x66,0x6C,0x78,0x00,0x00],
        'E' => [0x7E,0x60,0x60,0x60,0x78,0x60,0x60,0x60,0x60,0x7E,0x00,0x00],
        'F' => [0x7E,0x60,0x60,0x60,0x78,0x60,0x60,0x60,0x60,0x60,0x00,0x00],
        'G' => [0x3C,0x66,0x60,0x60,0x6E,0x66,0x66,0x66,0x66,0x3C,0x00,0x00],
        'H' => [0x66,0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x66,0x66,0x00,0x00],
        'I' => [0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00],
        'J' => [0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x4C,0x4C,0x38,0x00,0x00],
        'K' => [0x66,0x6C,0x78,0x70,0x70,0x78,0x6C,0x66,0x66,0x66,0x00,0x00],
        'L' => [0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x7E,0x00,0x00],
        'M' => [0x63,0x77,0x7F,0x6B,0x63,0x63,0x63,0x63,0x63,0x63,0x00,0x00],
        'N' => [0x66,0x76,0x7E,0x7E,0x6E,0x66,0x66,0x66,0x66,0x66,0x00,0x00],
        'O' => [0x3C,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x00],
        'P' => [0x7C,0x66,0x66,0x66,0x7C,0x60,0x60,0x60,0x60,0x60,0x00,0x00],
        'Q' => [0x3C,0x66,0x66,0x66,0x66,0x66,0x66,0x6E,0x3C,0x0E,0x00,0x00],
        'R' => [0x7C,0x66,0x66,0x66,0x7C,0x78,0x6C,0x66,0x66,0x66,0x00,0x00],
        'S' => [0x3C,0x66,0x60,0x60,0x3C,0x06,0x06,0x06,0x66,0x3C,0x00,0x00],
        'T' => [0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00],
        'U' => [0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x00],
        'V' => [0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x3C,0x18,0x00,0x00],
        'W' => [0x63,0x63,0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x63,0x00,0x00],
        'X' => [0x66,0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x66,0x66,0x00,0x00],
        'Y' => [0x66,0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x18,0x18,0x00,0x00],
        'Z' => [0x7E,0x06,0x06,0x0C,0x18,0x30,0x60,0x60,0x60,0x7E,0x00,0x00],
        'a' => [0x00,0x00,0x00,0x3C,0x06,0x3E,0x66,0x66,0x66,0x3E,0x00,0x00],
        'b' => [0x60,0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x66,0x7C,0x00,0x00],
        'c' => [0x00,0x00,0x00,0x3C,0x66,0x60,0x60,0x60,0x66,0x3C,0x00,0x00],
        'd' => [0x06,0x06,0x06,0x3E,0x66,0x66,0x66,0x66,0x66,0x3E,0x00,0x00],
        'e' => [0x00,0x00,0x00,0x3C,0x66,0x7E,0x60,0x60,0x66,0x3C,0x00,0x00],
        'f' => [0x1C,0x36,0x30,0x30,0x7C,0x30,0x30,0x30,0x30,0x30,0x00,0x00],
        'g' => [0x00,0x00,0x00,0x3E,0x66,0x66,0x66,0x3E,0x06,0x66,0x3C,0x00],
        'h' => [0x60,0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x66,0x66,0x00,0x00],
        'i' => [0x18,0x00,0x00,0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00],
        'j' => [0x0C,0x00,0x00,0x1C,0x0C,0x0C,0x0C,0x0C,0x4C,0x4C,0x38,0x00],
        'k' => [0x60,0x60,0x60,0x66,0x6C,0x78,0x78,0x6C,0x66,0x66,0x00,0x00],
        'l' => [0x38,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00],
        'm' => [0x00,0x00,0x00,0x76,0x7F,0x6B,0x6B,0x63,0x63,0x63,0x00,0x00],
        'n' => [0x00,0x00,0x00,0x7C,0x66,0x66,0x66,0x66,0x66,0x66,0x00,0x00],
        'o' => [0x00,0x00,0x00,0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x00],
        'p' => [0x00,0x00,0x00,0x7C,0x66,0x66,0x66,0x7C,0x60,0x60,0x60,0x00],
        'q' => [0x00,0x00,0x00,0x3E,0x66,0x66,0x66,0x3E,0x06,0x06,0x06,0x00],
        'r' => [0x00,0x00,0x00,0x7C,0x66,0x60,0x60,0x60,0x60,0x60,0x00,0x00],
        's' => [0x00,0x00,0x00,0x3E,0x60,0x60,0x3C,0x06,0x06,0x7C,0x00,0x00],
        't' => [0x00,0x30,0x30,0x7C,0x30,0x30,0x30,0x30,0x36,0x1C,0x00,0x00],
        'u' => [0x00,0x00,0x00,0x66,0x66,0x66,0x66,0x66,0x66,0x3E,0x00,0x00],
        'v' => [0x00,0x00,0x00,0x66,0x66,0x66,0x66,0x3C,0x3C,0x18,0x00,0x00],
        'w' => [0x00,0x00,0x00,0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00,0x00],
        'x' => [0x00,0x00,0x00,0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x00,0x00],
        'y' => [0x00,0x00,0x00,0x66,0x66,0x66,0x66,0x3E,0x06,0x66,0x3C,0x00],
        'z' => [0x00,0x00,0x00,0x7E,0x0C,0x18,0x30,0x60,0x60,0x7E,0x00,0x00],
        '%' => [0x62,0x66,0x0C,0x0C,0x18,0x18,0x30,0x30,0x66,0x46,0x00,0x00],
        '.' => [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00],
        ':' => [0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00],
        ' ' => [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '/' => [0x02,0x06,0x0C,0x0C,0x18,0x18,0x30,0x30,0x60,0x40,0x00,0x00],
        '-' => [0x00,0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00],
        '+' => [0x00,0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00,0x00,0x00],
        '#' => [0x00,0x24,0x24,0x7E,0x24,0x24,0x7E,0x24,0x24,0x00,0x00,0x00],
        '(' => [0x0C,0x18,0x30,0x30,0x30,0x30,0x30,0x30,0x18,0x0C,0x00,0x00],
        ')' => [0x30,0x18,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x18,0x30,0x00,0x00],
        _ => [0xFF,0x81,0x81,0x81,0x81,0x81,0x81,0x81,0x81,0xFF,0x00,0x00],
    }
}

fn atlas_dimensions() -> (u32, u32) {
    let n = CHARSET.len() as u32;
    let rows = (n + GLYPHS_PER_ROW - 1) / GLYPHS_PER_ROW;
    (GLYPHS_PER_ROW * GLYPH_W, rows * GLYPH_H)
}

fn generate_atlas_pixels() -> (Vec<u8>, u32, u32) {
    let (w, h) = atlas_dimensions();
    let mut pixels = vec![0u8; (w * h) as usize];

    for (ci, ch) in CHARSET.chars().enumerate() {
        let bits = glyph_bits(ch);
        let col = (ci as u32) % GLYPHS_PER_ROW;
        let row = (ci as u32) / GLYPHS_PER_ROW;
        let ox = col * GLYPH_W;
        let oy = row * GLYPH_H;

        for (gy, &row_bits) in bits.iter().enumerate() {
            for gx in 0..8u32 {
                let lit = (row_bits >> (7 - gx)) & 1;
                let px = ox + gx;
                let py = oy + gy as u32;
                pixels[(py * w + px) as usize] = if lit != 0 { 255 } else { 0 };
            }
        }
    }

    (pixels, w, h)
}

/// GPU vertex for text rendering: position in NDC, UV coordinates into the
/// font atlas, and per-character RGBA color.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl TextVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        static ATTRS: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 20,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}

pub const TEXT_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@group(0) @binding(0)
var font_texture: texture_2d<f32>;
@group(0) @binding(1)
var font_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let alpha = textureSample(font_texture, font_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

/// Renders bitmap text as textured quads using a procedurally generated font atlas.
///
/// The atlas is an R8Unorm texture containing 8×12 pixel glyphs for ASCII characters.
/// Text is rendered as screen-space quads with alpha blending — each character maps
/// to a UV rectangle in the atlas. The WGSL shader samples the red channel as alpha.
pub struct TextRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    atlas_w: u32,
    atlas_h: u32,
}

impl TextRenderer {
    /// Create the text renderer: generates the font atlas texture, creates the
    /// render pipeline with alpha blending, and sets up the bind group.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) -> Self {
        let (pixels, atlas_w, atlas_h) = generate_atlas_pixels();

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("font_atlas"),
                size: wgpu::Extent3d {
                    width: atlas_w,
                    height: atlas_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &pixels,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("font_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("text_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text_shader"),
            source: wgpu::ShaderSource::Wgsl(TEXT_SHADER_SRC.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("text_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[TextVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
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
                module: &shader,
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

        Self { pipeline, bind_group, atlas_w, atlas_h }
    }

    /// Generate vertex/index data for a single text string at screen position (x, y) in NDC.
    /// `scale` controls glyph height in NDC units. Returns (vertices, indices) for GPU upload.
    pub fn build_text(
        &self,
        text: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
    ) -> (Vec<TextVertex>, Vec<u32>) {
        let glyph_ndc_h = scale;
        let glyph_ndc_w = scale * (GLYPH_W as f32 / GLYPH_H as f32);

        let atlas_cols = GLYPHS_PER_ROW;
        let u_step = GLYPH_W as f32 / self.atlas_w as f32;
        let v_step = GLYPH_H as f32 / self.atlas_h as f32;

        let mut verts = Vec::with_capacity(text.len() * 4);
        let mut idxs = Vec::with_capacity(text.len() * 6);
        let mut cursor_x = x;

        for ch in text.chars() {
            let ci = match char_index(ch) {
                Some(i) => i,
                None => continue,
            };

            let col = (ci as u32) % atlas_cols;
            let row = (ci as u32) / atlas_cols;

            let u0 = col as f32 * u_step;
            let v0 = row as f32 * v_step;
            let u1 = u0 + u_step;
            let v1 = v0 + v_step;

            let x0 = cursor_x;
            let x1 = cursor_x + glyph_ndc_w;
            let y0 = y;
            let y1 = y - glyph_ndc_h;

            let base = verts.len() as u32;
            verts.push(TextVertex { position: [x0, y0, 0.0], uv: [u0, v0], color });
            verts.push(TextVertex { position: [x1, y0, 0.0], uv: [u1, v0], color });
            verts.push(TextVertex { position: [x1, y1, 0.0], uv: [u1, v1], color });
            verts.push(TextVertex { position: [x0, y1, 0.0], uv: [u0, v1], color });
            idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

            cursor_x = x1;
        }

        (verts, idxs)
    }

    /// Batch-generate vertex/index data for multiple text items into a single draw call.
    pub fn build_texts(
        &self,
        items: &[TextItem],
    ) -> (Vec<TextVertex>, Vec<u32>) {
        let mut all_verts: Vec<TextVertex> = Vec::new();
        let mut all_idxs: Vec<u32> = Vec::new();

        for item in items {
            let (verts, idxs) = self.build_text(&item.text, item.x, item.y, item.scale, item.color);
            let base = all_verts.len() as u32;
            all_verts.extend_from_slice(&verts);
            all_idxs.extend(idxs.iter().map(|i| i + base));
        }

        (all_verts, all_idxs)
    }
}

/// A text string with position, scale, and color — input to batch text rendering.
#[derive(Clone)]
pub struct TextItem {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub color: [f32; 4],
}
