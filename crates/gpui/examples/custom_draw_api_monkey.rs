//! Custom Draw API (3D Monkey) Example
//!
//! Demonstrates rendering Blender's Suzanne mesh with depth testing into an
//! offscreen target, then compositing that target into the GPUI window.

#![cfg_attr(target_family = "wasm", no_main)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, anyhow};
use gpui::colors::Colors;
use gpui::{
    App, AppContext, Bounds, Context, CustomAddressMode, CustomBindingDesc, CustomBindingKind,
    CustomBindingName, CustomBindingValue, CustomBlendMode, CustomBufferDesc, CustomBufferId,
    CustomBufferSource, CustomCullMode, CustomDepthCompare, CustomDepthFormat, CustomDepthState,
    CustomDepthTargetDesc, CustomDepthTargetId, CustomDrawParams, CustomFilterMode,
    CustomIndexBuffer, CustomIndexFormat, CustomPipelineDesc, CustomPipelineId,
    CustomPipelineState, CustomPrimitiveTopology, CustomRenderTarget, CustomRenderTargetDesc,
    CustomSamplerDesc, CustomSamplerId, CustomTextureFormat, CustomTextureId, CustomUniformBuilder,
    CustomVertexAttribute, CustomVertexAttributeName, CustomVertexBuffer, CustomVertexFetch,
    CustomVertexFormat, CustomVertexLayout, Hsla, Render, Styled, Window, WindowBounds,
    WindowOptions, canvas, div, prelude::*, px, size,
};
use gpui_platform::application;

const MONKEY_OBJ_SOURCE: &str = include_str!("assets/suzanne.obj");
const OFFSCREEN_TARGET_SIZE: u32 = 512;
const OFFSCREEN_SAMPLE_COUNT: u32 = 4;
const MONKEY_UNIFORM_SIZE: u32 = 144;

const MONKEY_SHADER_SOURCE: &str = r#"
struct VertexInput {
  a0: vec3<f32>,
  a1: vec3<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_normal: vec3<f32>,
};

struct SceneUniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  light_dir: vec4<f32>,
};

var<uniform> b0: SceneUniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.position = b0.mvp * vec4<f32>(input.a0, 1.0);
  out.world_normal = normalize((b0.model * vec4<f32>(input.a1, 0.0)).xyz);
  return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let light = normalize(b0.light_dir.xyz);
  let normal = normalize(input.world_normal);
  let diffuse = max(dot(normal, light), 0.0);
  let view = vec3<f32>(0.0, 0.0, 1.0);
  let half_vec = normalize(light + view);
  let specular = pow(max(dot(normal, half_vec), 0.0), 32.0);
  let base = vec3<f32>(0.76, 0.73, 0.69);
  let shaded = base * (0.22 + 0.78 * diffuse) + vec3<f32>(0.3 * specular);
  return vec4<f32>(shaded, 1.0);
}
"#;

const BLIT_SHADER_SOURCE: &str = r#"
struct VertexInput {
  a0: vec2<f32>,
  a1: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

var b0: texture_2d<f32>;
var b1: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4<f32>(input.a0, 0.0, 1.0);
  out.uv = input.a1;
  return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(b0, b1, input.uv);
}
"#;

type Mat4 = [f32; 16];

struct ParsedMesh {
    vertex_data: Arc<[u8]>,
    index_data: Arc<[u8]>,
    index_count: u32,
    center: [f32; 3],
    radius: f32,
}

struct MonkeyCustomDrawExample {
    monkey_pipeline: Option<CustomPipelineId>,
    blit_pipeline: Option<CustomPipelineId>,
    monkey_vertex_buffer: Option<CustomBufferId>,
    monkey_index_buffer: Option<CustomBufferId>,
    blit_vertex_buffer: Option<CustomBufferId>,
    render_target: Option<CustomTextureId>,
    depth_target: Option<CustomDepthTargetId>,
    sampler: Option<CustomSamplerId>,
    monkey_index_count: u32,
    monkey_center: [f32; 3],
    monkey_radius: f32,
    start: Instant,
    error: Option<String>,
}

impl MonkeyCustomDrawExample {
    fn new(_cx: &mut Context<Self>) -> Self {
        Self {
            monkey_pipeline: None,
            blit_pipeline: None,
            monkey_vertex_buffer: None,
            monkey_index_buffer: None,
            blit_vertex_buffer: None,
            render_target: None,
            depth_target: None,
            sampler: None,
            monkey_index_count: 0,
            monkey_center: [0.0, 0.0, 0.0],
            monkey_radius: 1.0,
            start: Instant::now(),
            error: None,
        }
    }

    fn ensure_resources(&mut self, window: &mut Window) {
        if self.monkey_pipeline.is_some() || self.error.is_some() {
            return;
        }

        match self.build_resources(window) {
            Ok(resources) => {
                self.monkey_pipeline = Some(resources.monkey_pipeline);
                self.blit_pipeline = Some(resources.blit_pipeline);
                self.monkey_vertex_buffer = Some(resources.monkey_vertex_buffer);
                self.monkey_index_buffer = Some(resources.monkey_index_buffer);
                self.blit_vertex_buffer = Some(resources.blit_vertex_buffer);
                self.render_target = Some(resources.render_target);
                self.depth_target = Some(resources.depth_target);
                self.sampler = Some(resources.sampler);
                self.monkey_index_count = resources.monkey_index_count;
                self.monkey_center = resources.monkey_center;
                self.monkey_radius = resources.monkey_radius;
            }
            Err(error) => {
                self.error = Some(error.to_string());
            }
        }
    }

    fn build_resources(&self, window: &mut Window) -> anyhow::Result<MonkeyResources> {
        let mesh = parse_obj_mesh(MONKEY_OBJ_SOURCE)?;

        let render_target = window.create_custom_render_target(CustomRenderTargetDesc {
            name: "custom_draw_monkey_color".to_string(),
            width: OFFSCREEN_TARGET_SIZE,
            height: OFFSCREEN_TARGET_SIZE,
            format: CustomTextureFormat::Rgba8Unorm,
            sample_count: OFFSCREEN_SAMPLE_COUNT,
            clear_color: Some([0.06, 0.07, 0.09, 1.0]),
        })?;

        let depth_target = window.create_custom_depth_target(CustomDepthTargetDesc {
            name: "custom_draw_monkey_depth".to_string(),
            width: OFFSCREEN_TARGET_SIZE,
            height: OFFSCREEN_TARGET_SIZE,
            format: CustomDepthFormat::Depth32Float,
            sample_count: OFFSCREEN_SAMPLE_COUNT,
            clear_depth: Some(1.0),
        })?;

        let monkey_pipeline = window.create_custom_pipeline(CustomPipelineDesc {
            name: "custom_draw_monkey_pipeline".to_string(),
            shader_source: MONKEY_SHADER_SOURCE.to_string(),
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            vertex_fetches: vec![CustomVertexFetch {
                layout: CustomVertexLayout {
                    stride: 24,
                    attributes: vec![
                        CustomVertexAttribute {
                            name: CustomVertexAttributeName::A0,
                            offset: 0,
                            format: CustomVertexFormat::F32Vec3,
                            location: None,
                        },
                        CustomVertexAttribute {
                            name: CustomVertexAttributeName::A1,
                            offset: 12,
                            format: CustomVertexFormat::F32Vec3,
                            location: None,
                        },
                    ],
                },
                instanced: false,
            }],
            primitive: CustomPrimitiveTopology::TriangleList,
            color_targets: vec![CustomTextureFormat::Rgba8Unorm],
            state: CustomPipelineState {
                blend: CustomBlendMode::Opaque,
                cull_mode: CustomCullMode::Back,
                depth: Some(CustomDepthState {
                    format: CustomDepthFormat::Depth32Float,
                    compare: CustomDepthCompare::LessEqual,
                    write_enabled: true,
                }),
                sample_count: OFFSCREEN_SAMPLE_COUNT,
                ..CustomPipelineState::default()
            },
            push_constants: None,
            bindings: vec![CustomBindingDesc {
                name: CustomBindingName::B0,
                kind: CustomBindingKind::Uniform {
                    size: MONKEY_UNIFORM_SIZE,
                },
                slot: None,
            }],
        })?;

        let blit_pipeline = window.create_custom_pipeline(CustomPipelineDesc {
            name: "custom_draw_monkey_blit_pipeline".to_string(),
            shader_source: BLIT_SHADER_SOURCE.to_string(),
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            vertex_fetches: vec![CustomVertexFetch {
                layout: CustomVertexLayout {
                    stride: 16,
                    attributes: vec![
                        CustomVertexAttribute {
                            name: CustomVertexAttributeName::A0,
                            offset: 0,
                            format: CustomVertexFormat::F32Vec2,
                            location: None,
                        },
                        CustomVertexAttribute {
                            name: CustomVertexAttributeName::A1,
                            offset: 8,
                            format: CustomVertexFormat::F32Vec2,
                            location: None,
                        },
                    ],
                },
                instanced: false,
            }],
            primitive: CustomPrimitiveTopology::TriangleList,
            color_targets: Vec::new(),
            state: CustomPipelineState::default(),
            push_constants: None,
            bindings: vec![
                CustomBindingDesc {
                    name: CustomBindingName::B0,
                    kind: CustomBindingKind::Texture,
                    slot: None,
                },
                CustomBindingDesc {
                    name: CustomBindingName::B1,
                    kind: CustomBindingKind::Sampler,
                    slot: None,
                },
            ],
        })?;

        let monkey_vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "custom_draw_monkey_vertices".to_string(),
            data: Arc::clone(&mesh.vertex_data),
        })?;

        let monkey_index_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "custom_draw_monkey_indices".to_string(),
            data: Arc::clone(&mesh.index_data),
        })?;

        let blit_vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "custom_draw_monkey_blit_vertices".to_string(),
            data: blit_vertex_placeholder(),
        })?;

        let sampler = window.create_custom_sampler(CustomSamplerDesc {
            name: "custom_draw_monkey_sampler".to_string(),
            min_filter: CustomFilterMode::Linear,
            mag_filter: CustomFilterMode::Linear,
            mipmap_filter: CustomFilterMode::Nearest,
            address_modes: [CustomAddressMode::ClampToEdge; 3],
        })?;

        Ok(MonkeyResources {
            monkey_pipeline,
            blit_pipeline,
            monkey_vertex_buffer,
            monkey_index_buffer,
            blit_vertex_buffer,
            render_target,
            depth_target,
            sampler,
            monkey_index_count: mesh.index_count,
            monkey_center: mesh.center,
            monkey_radius: mesh.radius,
        })
    }
}

impl Render for MonkeyCustomDrawExample {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let colors = Colors::for_appearance(window);
        self.ensure_resources(window);
        window.request_animation_frame();

        let header = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xl()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(colors.text)
                    .child("Custom Draw API (3D Monkey)"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(colors.disabled)
                    .child("Blender Suzanne rendered offscreen with depth/MSAA"),
            );

        let surface: Hsla = colors.container.into();
        let content = if let Some(error) = &self.error {
            div()
                .text_sm()
                .text_color(gpui::red())
                .child(format!("Custom draw unsupported: {error}"))
        } else if let (
            Some(monkey_pipeline),
            Some(blit_pipeline),
            Some(monkey_vertex_buffer),
            Some(monkey_index_buffer),
            Some(blit_vertex_buffer),
            Some(render_target),
            Some(depth_target),
            Some(sampler),
        ) = (
            self.monkey_pipeline,
            self.blit_pipeline,
            self.monkey_vertex_buffer,
            self.monkey_index_buffer,
            self.blit_vertex_buffer,
            self.render_target,
            self.depth_target,
            self.sampler,
        ) {
            let start = self.start;
            let monkey_index_count = self.monkey_index_count;
            let monkey_center = self.monkey_center;
            let monkey_radius = self.monkey_radius;

            let prepaint = move |bounds: Bounds<_>, window: &mut Window, _cx: &mut App| {
                let blit_vertices = blit_vertex_data_for_bounds(bounds, window.viewport_size());
                if let Err(error) =
                    window.update_custom_buffer(blit_vertex_buffer, Arc::clone(&blit_vertices))
                {
                    log::error!("custom draw blit vertex update failed: {error}");
                }

                let elapsed = start.elapsed().as_secs_f32();
                let scene_uniform = build_scene_uniform(elapsed, monkey_center, monkey_radius);
                let offscreen_target = CustomRenderTarget {
                    colors: vec![render_target],
                    depth: Some(depth_target),
                };

                vec![
                    CustomDrawParams {
                        bounds,
                        pipeline: monkey_pipeline,
                        vertex_buffers: vec![CustomVertexBuffer {
                            source: CustomBufferSource::Buffer(monkey_vertex_buffer),
                        }],
                        vertex_count: 0,
                        index_buffer: Some(CustomIndexBuffer {
                            source: CustomBufferSource::Buffer(monkey_index_buffer),
                            format: CustomIndexFormat::U32,
                        }),
                        index_count: monkey_index_count,
                        target: Some(offscreen_target),
                        instance_count: 1,
                        push_constants: None,
                        bindings: vec![CustomBindingValue::Uniform(CustomBufferSource::Inline(
                            scene_uniform,
                        ))],
                    },
                    CustomDrawParams {
                        bounds,
                        pipeline: blit_pipeline,
                        vertex_buffers: vec![CustomVertexBuffer {
                            source: CustomBufferSource::Buffer(blit_vertex_buffer),
                        }],
                        vertex_count: 6,
                        index_buffer: None,
                        index_count: 0,
                        target: None,
                        instance_count: 1,
                        push_constants: None,
                        bindings: vec![
                            CustomBindingValue::Texture(render_target),
                            CustomBindingValue::Sampler(sampler),
                        ],
                    },
                ]
            };

            let paint = move |_bounds: Bounds<_>,
                              params: Vec<CustomDrawParams>,
                              window: &mut Window,
                              _cx: &mut App| {
                for draw in params {
                    if let Err(error) = window.paint_custom(draw) {
                        log::error!("custom draw paint failed: {error}");
                    }
                }
            };

            div()
                .w(px(480.))
                .h(px(480.))
                .rounded_md()
                .border_1()
                .border_color(colors.border)
                .bg(surface.opacity(0.2))
                .child(canvas(prepaint, paint).size_full())
        } else {
            div()
                .text_sm()
                .text_color(colors.disabled)
                .child("Initializing custom draw resources...")
        };

        div()
            .size_full()
            .p_6()
            .bg(colors.background)
            .child(div().flex().flex_col().gap_4().child(header).child(content))
    }
}

struct MonkeyResources {
    monkey_pipeline: CustomPipelineId,
    blit_pipeline: CustomPipelineId,
    monkey_vertex_buffer: CustomBufferId,
    monkey_index_buffer: CustomBufferId,
    blit_vertex_buffer: CustomBufferId,
    render_target: CustomTextureId,
    depth_target: CustomDepthTargetId,
    sampler: CustomSamplerId,
    monkey_index_count: u32,
    monkey_center: [f32; 3],
    monkey_radius: f32,
}

fn parse_obj_mesh(source: &str) -> anyhow::Result<ParsedMesh> {
    let mut positions = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut faces = Vec::<(usize, Vec<String>)>::new();

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("v ") {
            let position = parse_vec3(rest)
                .with_context(|| format!("failed to parse OBJ vertex at line {line_number}"))?;
            min[0] = min[0].min(position[0]);
            min[1] = min[1].min(position[1]);
            min[2] = min[2].min(position[2]);
            max[0] = max[0].max(position[0]);
            max[1] = max[1].max(position[1]);
            max[2] = max[2].max(position[2]);
            positions.push(position);
            continue;
        }

        if let Some(rest) = line.strip_prefix("vn ") {
            let normal = parse_vec3(rest)
                .with_context(|| format!("failed to parse OBJ normal at line {line_number}"))?;
            normals.push(normalize3(normal));
            continue;
        }

        if let Some(rest) = line.strip_prefix("f ") {
            let tokens = rest
                .split_whitespace()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            if tokens.len() < 3 {
                return Err(anyhow!(
                    "OBJ face at line {} has fewer than three vertices",
                    line_number
                ));
            }
            faces.push((line_number, tokens));
        }
    }

    if positions.is_empty() {
        return Err(anyhow!("OBJ mesh has no vertices"));
    }

    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];

    let radius = positions
        .iter()
        .map(|position| {
            let offset = [
                position[0] - center[0],
                position[1] - center[1],
                position[2] - center[2],
            ];
            (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt()
        })
        .fold(0.0f32, f32::max)
        .max(0.001);

    let mut vertex_lookup = HashMap::<(usize, usize), u32>::new();
    let mut vertex_data = Vec::<u8>::new();
    let mut indices = Vec::<u32>::new();

    for (line_number, tokens) in faces {
        let corners = tokens
            .iter()
            .map(|token| parse_face_vertex(token, positions.len(), normals.len(), line_number))
            .collect::<anyhow::Result<Vec<_>>>()?;

        for triangle in 1..corners.len() - 1 {
            let triangle_corners = [corners[0], corners[triangle], corners[triangle + 1]];
            for corner in triangle_corners {
                let normal_key = corner.1.unwrap_or(usize::MAX);
                let key = (corner.0, normal_key);
                let index = if let Some(index) = vertex_lookup.get(&key) {
                    *index
                } else {
                    let index = u32::try_from(vertex_lookup.len()).map_err(|_| {
                        anyhow!("OBJ mesh has too many unique vertices for u32 indices")
                    })?;
                    let position = positions[corner.0];
                    let normal = corner
                        .1
                        .map(|normal_index| normals[normal_index])
                        .unwrap_or([0.0, 1.0, 0.0]);
                    push_f32(&mut vertex_data, position[0]);
                    push_f32(&mut vertex_data, position[1]);
                    push_f32(&mut vertex_data, position[2]);
                    push_f32(&mut vertex_data, normal[0]);
                    push_f32(&mut vertex_data, normal[1]);
                    push_f32(&mut vertex_data, normal[2]);
                    vertex_lookup.insert(key, index);
                    index
                };
                indices.push(index);
            }
        }
    }

    if indices.is_empty() {
        return Err(anyhow!("OBJ mesh has no faces"));
    }

    let index_count = u32::try_from(indices.len())
        .map_err(|_| anyhow!("OBJ mesh index count exceeds u32::MAX"))?;

    let mut index_data = Vec::with_capacity(indices.len() * std::mem::size_of::<u32>());
    for index in indices {
        index_data.extend_from_slice(&index.to_le_bytes());
    }

    Ok(ParsedMesh {
        vertex_data: Arc::from(vertex_data),
        index_data: Arc::from(index_data),
        index_count,
        center,
        radius,
    })
}

fn parse_vec3(input: &str) -> anyhow::Result<[f32; 3]> {
    let values = input
        .split_whitespace()
        .take(3)
        .map(|value| value.parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .context("failed to parse float")?;

    if values.len() != 3 {
        return Err(anyhow!("expected three float components"));
    }

    Ok([values[0], values[1], values[2]])
}

fn parse_face_vertex(
    token: &str,
    position_count: usize,
    normal_count: usize,
    line_number: usize,
) -> anyhow::Result<(usize, Option<usize>)> {
    let parts = token.split('/').collect::<Vec<_>>();
    if parts.is_empty() || parts[0].is_empty() {
        return Err(anyhow!(
            "OBJ face at line {} has an empty vertex reference",
            line_number
        ));
    }

    let position_index = parse_obj_index(parts[0], position_count, line_number)?;
    let normal_index = if parts.len() >= 3 && !parts[2].is_empty() {
        Some(parse_obj_index(parts[2], normal_count, line_number)?)
    } else {
        None
    };

    Ok((position_index, normal_index))
}

fn parse_obj_index(raw: &str, count: usize, line_number: usize) -> anyhow::Result<usize> {
    if count == 0 {
        return Err(anyhow!(
            "OBJ face at line {} references an empty attribute list",
            line_number
        ));
    }

    let parsed = raw.parse::<i32>().with_context(|| {
        format!(
            "OBJ face at line {} contains invalid index '{}': expected integer",
            line_number, raw
        )
    })?;

    if parsed == 0 {
        return Err(anyhow!(
            "OBJ face at line {} uses index 0, but OBJ indices are 1-based",
            line_number
        ));
    }

    let resolved = if parsed > 0 {
        parsed - 1
    } else {
        i32::try_from(count).map_err(|_| anyhow!("attribute count exceeds i32::MAX"))? + parsed
    };

    if resolved < 0 {
        return Err(anyhow!(
            "OBJ face at line {} resolves to a negative index ({})",
            line_number,
            resolved
        ));
    }

    let resolved_index =
        usize::try_from(resolved).map_err(|_| anyhow!("resolved OBJ index conversion failed"))?;

    if resolved_index >= count {
        return Err(anyhow!(
            "OBJ face at line {} references index {} outside range 0..{}",
            line_number,
            resolved_index,
            count
        ));
    }

    Ok(resolved_index)
}

fn build_scene_uniform(time_seconds: f32, center: [f32; 3], radius: f32) -> Arc<[u8]> {
    let center_translation = mat4_translation([-center[0], -center[1], -center[2]]);
    let scale = mat4_scale(1.35 / radius.max(0.001));
    let rotation = mat4_mul(
        mat4_rotation_y(time_seconds * 0.9),
        mat4_rotation_x(-0.35 + 0.05 * (time_seconds * 0.7).sin()),
    );
    let model = mat4_mul(rotation, mat4_mul(scale, center_translation));

    let view = mat4_look_at([0.0, 0.2, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let projection = mat4_perspective(50f32.to_radians(), 1.0, 0.1, 50.0);
    let mvp = mat4_mul(projection, mat4_mul(view, model));

    let light_direction = normalize3([0.4, 0.8, 0.6]);

    let mut builder = CustomUniformBuilder::new();
    builder.push_mat4(mvp).push_mat4(model).push_vec4(
        light_direction[0],
        light_direction[1],
        light_direction[2],
        0.0,
    );
    builder.finish()
}

fn blit_vertex_placeholder() -> Arc<[u8]> {
    Arc::from(vec![0u8; 6 * 4 * 4])
}

fn blit_vertex_data_for_bounds(
    bounds: Bounds<gpui::Pixels>,
    viewport: gpui::Size<gpui::Pixels>,
) -> Arc<[u8]> {
    let left = bounds.origin.x;
    let top = bounds.origin.y;
    let right = bounds.origin.x + bounds.size.width;
    let bottom = bounds.origin.y + bounds.size.height;

    let viewport_width = f32::from(viewport.width).max(1.0);
    let viewport_height = f32::from(viewport.height).max(1.0);

    let to_ndc_x = |x: gpui::Pixels| (f32::from(x) / viewport_width) * 2.0 - 1.0;
    let to_ndc_y = |y: gpui::Pixels| 1.0 - (f32::from(y) / viewport_height) * 2.0;

    let left_ndc = to_ndc_x(left);
    let right_ndc = to_ndc_x(right);
    let top_ndc = to_ndc_y(top);
    let bottom_ndc = to_ndc_y(bottom);

    let vertices = [
        (left_ndc, top_ndc, 0.0, 0.0),
        (right_ndc, top_ndc, 1.0, 0.0),
        (right_ndc, bottom_ndc, 1.0, 1.0),
        (left_ndc, top_ndc, 0.0, 0.0),
        (right_ndc, bottom_ndc, 1.0, 1.0),
        (left_ndc, bottom_ndc, 0.0, 1.0),
    ];

    let mut data = Vec::with_capacity(vertices.len() * 16);
    for (x, y, u, v) in vertices {
        push_f32(&mut data, x);
        push_f32(&mut data, y);
        push_f32(&mut data, u);
        push_f32(&mut data, v);
    }

    Arc::from(data)
}

fn push_f32(data: &mut Vec<u8>, value: f32) {
    data.extend_from_slice(&value.to_le_bytes());
}

fn mat4_mul(left: Mat4, right: Mat4) -> Mat4 {
    let mut result = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            result[column * 4 + row] = left[row] * right[column * 4]
                + left[4 + row] * right[column * 4 + 1]
                + left[8 + row] * right[column * 4 + 2]
                + left[12 + row] * right[column * 4 + 3];
        }
    }
    result
}

fn mat4_translation(translation: [f32; 3]) -> Mat4 {
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ]
}

fn mat4_scale(scale: f32) -> Mat4 {
    [
        scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn mat4_rotation_x(angle: f32) -> Mat4 {
    let sine = angle.sin();
    let cosine = angle.cos();
    [
        1.0, 0.0, 0.0, 0.0, 0.0, cosine, sine, 0.0, 0.0, -sine, cosine, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn mat4_rotation_y(angle: f32) -> Mat4 {
    let sine = angle.sin();
    let cosine = angle.cos();
    [
        cosine, 0.0, -sine, 0.0, 0.0, 1.0, 0.0, 0.0, sine, 0.0, cosine, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn mat4_perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let focal_length = 1.0 / (fov_y_radians * 0.5).tan();
    let depth = near - far;

    [
        focal_length / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        focal_length,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) / depth,
        -1.0,
        0.0,
        0.0,
        (2.0 * far * near) / depth,
        0.0,
    ]
}

fn mat4_look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let forward = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let side = normalize3(cross3(forward, up));
    let up_direction = cross3(side, forward);

    [
        side[0],
        up_direction[0],
        -forward[0],
        0.0,
        side[1],
        up_direction[1],
        -forward[1],
        0.0,
        side[2],
        up_direction[2],
        -forward[2],
        0.0,
        -dot3(side, eye),
        -dot3(up_direction, eye),
        dot3(forward, eye),
        1.0,
    ]
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize3(vector: [f32; 3]) -> [f32; 3] {
    let length_squared = dot3(vector, vector);
    if length_squared <= f32::EPSILON {
        return [0.0, 1.0, 0.0];
    }
    let inverse_length = length_squared.sqrt().recip();
    [
        vector[0] * inverse_length,
        vector[1] * inverse_length,
        vector[2] * inverse_length,
    ]
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(580.), px(580.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(MonkeyCustomDrawExample::new),
        )
        .expect("failed to open window");

        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    run_example();
}
