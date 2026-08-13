#![cfg_attr(target_family = "wasm", no_main)]

use std::sync::Arc;

use gpui::colors::Colors;
use gpui::{
    App, AppContext, Bounds, Context, CustomAddressMode, CustomBindingDesc, CustomBindingKind,
    CustomBindingName, CustomBindingValue, CustomBlendMode, CustomBufferDesc, CustomBufferId,
    CustomBufferSource, CustomDrawParams, CustomFilterMode, CustomPipelineDesc, CustomPipelineId,
    CustomPipelineState, CustomPrimitiveTopology, CustomSamplerDesc, CustomSamplerId,
    CustomTextureDesc, CustomTextureDimension, CustomTextureFormat, CustomTextureId,
    CustomTextureUsage, CustomVertexAttribute, CustomVertexAttributeName, CustomVertexBuffer,
    CustomVertexFetch, CustomVertexFormat, CustomVertexLayout, Hsla, Render, Styled, Window,
    WindowBounds, WindowOptions, canvas, div, prelude::*, px, size,
};
use gpui_platform::application;

const CARD_WIDTH: f32 = 300.0;
const CARD_HEIGHT: f32 = 220.0;
const CARD_CORNER_RADIUS: f32 = 12.0;
const DEMO_TEXTURE_SIZE: u32 = 128;

const SHADER_SOURCE: &str = r#"
struct VertexInput {
  a0: vec2<f32>,
  a1: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct RoundedClipUniform {
  rect_size: vec2<f32>,
  corner_radius: f32,
  clip_enabled: f32,
};

var b0: texture_2d<f32>;
var b1: sampler;
var<uniform> b2: RoundedClipUniform;

fn rounded_rect_alpha(local_position: vec2<f32>, rect_size: vec2<f32>, corner_radius: f32) -> f32 {
  let half_size = rect_size * 0.5;
  let limited_corner_radius = clamp(corner_radius, 0.0, min(half_size.x, half_size.y));
  let centered_position = local_position - half_size;
  let q = abs(centered_position)
    - (half_size - vec2<f32>(limited_corner_radius, limited_corner_radius));
  let signed_distance = length(max(q, vec2<f32>(0.0, 0.0)))
    + min(max(q.x, q.y), 0.0)
    - limited_corner_radius;
  let antialias_pixels = max(fwidth(signed_distance), 0.75);
  return 1.0 - smoothstep(0.0, antialias_pixels, signed_distance);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4<f32>(input.a0, 0.0, 1.0);
  out.uv = input.a1;
  return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let base_color = textureSample(b0, b1, input.uv);
  if b2.clip_enabled < 0.5 {
    return base_color;
  }

  let local_position = input.uv * b2.rect_size;
  let alpha = rounded_rect_alpha(local_position, b2.rect_size, b2.corner_radius);
  return vec4<f32>(base_color.rgb, base_color.a * alpha);
}
"#;

struct RoundedClipExample {
    pipeline: Option<CustomPipelineId>,
    left_vertex_buffer: Option<CustomBufferId>,
    right_vertex_buffer: Option<CustomBufferId>,
    texture: Option<CustomTextureId>,
    sampler: Option<CustomSamplerId>,
    error: Option<String>,
}

impl RoundedClipExample {
    fn new(_cx: &mut Context<Self>) -> Self {
        Self {
            pipeline: None,
            left_vertex_buffer: None,
            right_vertex_buffer: None,
            texture: None,
            sampler: None,
            error: None,
        }
    }

    fn ensure_resources(&mut self, window: &mut Window) {
        if self.pipeline.is_some() || self.error.is_some() {
            return;
        }

        match self.build_resources(window) {
            Ok((pipeline, left_vertex_buffer, right_vertex_buffer, texture, sampler)) => {
                self.pipeline = Some(pipeline);
                self.left_vertex_buffer = Some(left_vertex_buffer);
                self.right_vertex_buffer = Some(right_vertex_buffer);
                self.texture = Some(texture);
                self.sampler = Some(sampler);
            }
            Err(error) => {
                self.error = Some(error.to_string());
            }
        }
    }

    fn build_resources(
        &self,
        window: &mut Window,
    ) -> anyhow::Result<(
        CustomPipelineId,
        CustomBufferId,
        CustomBufferId,
        CustomTextureId,
        CustomSamplerId,
    )> {
        let pipeline = window.create_custom_pipeline(CustomPipelineDesc {
            name: "custom_draw_rounded_clip_demo".to_string(),
            shader_source: SHADER_SOURCE.to_string(),
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
            state: CustomPipelineState {
                blend: CustomBlendMode::Alpha,
                ..Default::default()
            },
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
                CustomBindingDesc {
                    name: CustomBindingName::B2,
                    kind: CustomBindingKind::Uniform { size: 16 },
                    slot: None,
                },
            ],
        })?;

        let left_vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "rounded_clip_left_vertices".to_string(),
            data: quad_vertex_data(),
        })?;

        let right_vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "rounded_clip_right_vertices".to_string(),
            data: quad_vertex_data(),
        })?;

        let texture = window.create_custom_texture(CustomTextureDesc {
            name: "rounded_clip_demo_texture".to_string(),
            dimension: CustomTextureDimension::D2,
            width: DEMO_TEXTURE_SIZE,
            height: DEMO_TEXTURE_SIZE,
            format: CustomTextureFormat::Rgba8Unorm,
            usage: CustomTextureUsage::SAMPLED,
            data: vec![demo_texture_data(DEMO_TEXTURE_SIZE, DEMO_TEXTURE_SIZE)],
        })?;

        let sampler = window.create_custom_sampler(CustomSamplerDesc {
            name: "rounded_clip_demo_sampler".to_string(),
            min_filter: CustomFilterMode::Linear,
            mag_filter: CustomFilterMode::Linear,
            mipmap_filter: CustomFilterMode::Nearest,
            address_modes: [CustomAddressMode::ClampToEdge; 3],
        })?;

        Ok((
            pipeline,
            left_vertex_buffer,
            right_vertex_buffer,
            texture,
            sampler,
        ))
    }
}

impl Render for RoundedClipExample {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let colors = Colors::for_appearance(window);
        self.ensure_resources(window);

        let header = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xl()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(colors.text)
                    .child("Custom Draw API (Rounded Clip Helper)"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(colors.disabled)
                    .child("Rounded corner clipping is opt-in for custom draw shaders"),
            );

        let surface_color: Hsla = colors.container.into();

        let content = if let Some(error) = &self.error {
            div()
                .text_sm()
                .text_color(gpui::red())
                .child(format!("Custom draw unsupported: {error}"))
        } else if let (
            Some(pipeline),
            Some(left_vertex_buffer),
            Some(right_vertex_buffer),
            Some(texture),
            Some(sampler),
        ) = (
            self.pipeline,
            self.left_vertex_buffer,
            self.right_vertex_buffer,
            self.texture,
            self.sampler,
        ) {
            let left_prepaint = move |bounds: Bounds<_>, window: &mut Window, _cx: &mut App| {
                let vertex_data = quad_vertex_data_for_bounds(bounds, window.viewport_size());
                if let Err(error) = window.update_custom_buffer(left_vertex_buffer, vertex_data) {
                    log::error!("left rounded-clip vertex update failed: {error}");
                }

                CustomDrawParams {
                    bounds,
                    pipeline,
                    vertex_buffers: vec![CustomVertexBuffer {
                        source: CustomBufferSource::Buffer(left_vertex_buffer),
                    }],
                    vertex_count: 6,
                    index_buffer: None,
                    index_count: 0,
                    target: None,
                    instance_count: 1,
                    push_constants: None,
                    bindings: vec![
                        CustomBindingValue::Texture(texture),
                        CustomBindingValue::Sampler(sampler),
                        CustomBindingValue::Uniform(CustomBufferSource::Inline(
                            rounded_clip_uniform_data(bounds, false),
                        )),
                    ],
                }
            };

            let left_paint = move |_bounds: Bounds<_>,
                                   params: CustomDrawParams,
                                   window: &mut Window,
                                   _cx: &mut App| {
                if let Err(error) = window.paint_custom(params) {
                    log::error!("left rounded-clip paint failed: {error}");
                }
            };

            let right_prepaint = move |bounds: Bounds<_>, window: &mut Window, _cx: &mut App| {
                let vertex_data = quad_vertex_data_for_bounds(bounds, window.viewport_size());
                if let Err(error) = window.update_custom_buffer(right_vertex_buffer, vertex_data) {
                    log::error!("right rounded-clip vertex update failed: {error}");
                }

                CustomDrawParams {
                    bounds,
                    pipeline,
                    vertex_buffers: vec![CustomVertexBuffer {
                        source: CustomBufferSource::Buffer(right_vertex_buffer),
                    }],
                    vertex_count: 6,
                    index_buffer: None,
                    index_count: 0,
                    target: None,
                    instance_count: 1,
                    push_constants: None,
                    bindings: vec![
                        CustomBindingValue::Texture(texture),
                        CustomBindingValue::Sampler(sampler),
                        CustomBindingValue::Uniform(CustomBufferSource::Inline(
                            rounded_clip_uniform_data(bounds, true),
                        )),
                    ],
                }
            };

            let right_paint = move |_bounds: Bounds<_>,
                                    params: CustomDrawParams,
                                    window: &mut Window,
                                    _cx: &mut App| {
                if let Err(error) = window.paint_custom(params) {
                    log::error!("right rounded-clip paint failed: {error}");
                }
            };

            div()
                .flex()
                .gap_6()
                .child(
                    div()
                        .w(px(CARD_WIDTH))
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_sm()
                                .text_color(colors.disabled)
                                .child("Default: rectangle clip"),
                        )
                        .child(
                            div()
                                .w(px(CARD_WIDTH))
                                .h(px(CARD_HEIGHT))
                                .rounded(px(CARD_CORNER_RADIUS))
                                .border_1()
                                .border_color(colors.border)
                                .bg(surface_color.opacity(0.2))
                                .child(canvas(left_prepaint, left_paint).size_full()),
                        ),
                )
                .child(
                    div()
                        .w(px(CARD_WIDTH))
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_sm()
                                .text_color(colors.disabled)
                                .child("Shader helper: rounded clip"),
                        )
                        .child(
                            div()
                                .w(px(CARD_WIDTH))
                                .h(px(CARD_HEIGHT))
                                .rounded(px(CARD_CORNER_RADIUS))
                                .border_1()
                                .border_color(colors.border)
                                .bg(surface_color.opacity(0.2))
                                .child(canvas(right_prepaint, right_paint).size_full()),
                        ),
                )
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

fn append_f32(output: &mut Vec<u8>, value: f32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn quad_vertex_data() -> Arc<[u8]> {
    let mut data = Vec::with_capacity(6 * 4 * 4);
    let vertices = [
        (-0.9, -0.9, 0.0, 1.0),
        (0.9, -0.9, 1.0, 1.0),
        (0.9, 0.9, 1.0, 0.0),
        (-0.9, -0.9, 0.0, 1.0),
        (0.9, 0.9, 1.0, 0.0),
        (-0.9, 0.9, 0.0, 0.0),
    ];

    for (x, y, u, v) in vertices {
        append_f32(&mut data, x);
        append_f32(&mut data, y);
        append_f32(&mut data, u);
        append_f32(&mut data, v);
    }

    Arc::from(data)
}

fn quad_vertex_data_for_bounds(
    bounds: Bounds<gpui::Pixels>,
    viewport: gpui::Size<gpui::Pixels>,
) -> Arc<[u8]> {
    let mut data = Vec::with_capacity(6 * 4 * 4);

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

    for (x, y, u, v) in vertices {
        append_f32(&mut data, x);
        append_f32(&mut data, y);
        append_f32(&mut data, u);
        append_f32(&mut data, v);
    }

    Arc::from(data)
}

fn rounded_clip_uniform_data(bounds: Bounds<gpui::Pixels>, clip_enabled: bool) -> Arc<[u8]> {
    let mut data = Vec::with_capacity(16);
    append_f32(&mut data, f32::from(bounds.size.width).max(1.0));
    append_f32(&mut data, f32::from(bounds.size.height).max(1.0));
    append_f32(&mut data, CARD_CORNER_RADIUS);
    append_f32(&mut data, if clip_enabled { 1.0 } else { 0.0 });
    Arc::from(data)
}

fn demo_texture_data(width: u32, height: u32) -> Arc<[u8]> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let x_ratio = if width <= 1 {
                0.0
            } else {
                x as f32 / (width - 1) as f32
            };
            let y_ratio = if height <= 1 {
                0.0
            } else {
                y as f32 / (height - 1) as f32
            };

            let red = (x_ratio * 255.0).round() as u8;
            let green = (y_ratio * 255.0).round() as u8;
            let stripe = if ((x / 8) + (y / 8)) % 2 == 0 {
                240
            } else {
                90
            };

            data.push(red);
            data.push(green);
            data.push(stripe);
            data.push(255);
        }
    }

    Arc::from(data)
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(760.0), px(440.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(RoundedClipExample::new),
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
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
