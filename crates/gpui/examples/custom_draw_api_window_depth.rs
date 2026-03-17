#![cfg_attr(target_family = "wasm", no_main)]

use std::sync::Arc;
use std::time::Instant;

use gpui::colors::Colors;
use gpui::{
    App, AppContext, Bounds, Context, CustomBindingDesc, CustomBindingKind, CustomBindingName,
    CustomBindingValue, CustomBufferDesc, CustomBufferId, CustomBufferSource, CustomCullMode,
    CustomDepthCompare, CustomDepthFormat, CustomDepthState, CustomDrawParams, CustomIndexBuffer,
    CustomIndexFormat, CustomPipelineDesc, CustomPipelineId, CustomPipelineState,
    CustomPrimitiveTopology, CustomUniformBuilder, CustomVertexAttribute,
    CustomVertexAttributeName, CustomVertexBuffer, CustomVertexFetch, CustomVertexFormat,
    CustomVertexLayout, Hsla, Render, Styled, Window, WindowBounds, WindowOptions, canvas, div,
    prelude::*, px, size,
};
use gpui_platform::application;

const SHADER_SOURCE: &str = r#"
struct VertexInput {
  a0: vec3<f32>,
  a1: vec3<f32>,
};

struct SceneUniforms {
  mvp: mat4x4<f32>,
  bounds: vec4<f32>,
  viewport: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

var<uniform> b0: SceneUniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;

  let local_clip = b0.mvp * vec4<f32>(input.a0, 1.0);
  let local_ndc = local_clip.xyz / local_clip.w;
  let local_uv = local_ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);

  let pixel = b0.bounds.xy + local_uv * b0.bounds.zw;
  let mapped_ndc = vec2<f32>(
    (pixel.x / b0.viewport.x) * 2.0 - 1.0,
    1.0 - (pixel.y / b0.viewport.y) * 2.0
  );

  out.position = vec4<f32>(mapped_ndc, local_ndc.z, 1.0);
  out.color = input.a1;
  return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color, 1.0);
}
"#;

const UNIFORM_SIZE: u32 = 96;
const INDEX_COUNT: u32 = 36;

type Mat4 = [f32; 16];

struct WindowDepthExample {
    pipeline: Option<CustomPipelineId>,
    vertex_buffer: Option<CustomBufferId>,
    index_buffer: Option<CustomBufferId>,
    error: Option<String>,
    start: Instant,
}

impl WindowDepthExample {
    fn new(_cx: &mut Context<Self>) -> Self {
        Self {
            pipeline: None,
            vertex_buffer: None,
            index_buffer: None,
            error: None,
            start: Instant::now(),
        }
    }

    fn ensure_resources(&mut self, window: &mut Window) {
        if self.pipeline.is_some() || self.error.is_some() {
            return;
        }

        match self.build_resources(window) {
            Ok((pipeline, vertex_buffer, index_buffer)) => {
                self.pipeline = Some(pipeline);
                self.vertex_buffer = Some(vertex_buffer);
                self.index_buffer = Some(index_buffer);
            }
            Err(error) => {
                self.error = Some(error.to_string());
            }
        }
    }

    fn build_resources(
        &self,
        window: &mut Window,
    ) -> anyhow::Result<(CustomPipelineId, CustomBufferId, CustomBufferId)> {
        let pipeline = window.create_custom_pipeline(CustomPipelineDesc {
            name: "custom_draw_window_depth".to_string(),
            shader_source: SHADER_SOURCE.to_string(),
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
            color_targets: Vec::new(),
            state: CustomPipelineState {
                cull_mode: CustomCullMode::Back,
                depth: Some(CustomDepthState {
                    format: CustomDepthFormat::Depth32Float,
                    compare: CustomDepthCompare::LessEqual,
                    write_enabled: true,
                }),
                ..CustomPipelineState::default()
            },
            push_constants: None,
            bindings: vec![CustomBindingDesc {
                name: CustomBindingName::B0,
                kind: CustomBindingKind::Uniform { size: UNIFORM_SIZE },
                slot: None,
            }],
        })?;

        let vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "custom_draw_window_depth_vertices".to_string(),
            data: cube_vertex_data(),
        })?;

        let index_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "custom_draw_window_depth_indices".to_string(),
            data: cube_index_data(),
        })?;

        Ok((pipeline, vertex_buffer, index_buffer))
    }
}

impl Render for WindowDepthExample {
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
                    .child("Custom Draw API (Window Depth)"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(colors.disabled)
                    .child("Direct window custom draw with depth-tested 3D cube"),
            );

        let surface: Hsla = colors.container.into();
        let content = if let Some(error) = &self.error {
            div()
                .text_sm()
                .text_color(gpui::red())
                .child(format!("Custom draw unsupported: {error}"))
        } else if let (Some(pipeline), Some(vertex_buffer), Some(index_buffer)) =
            (self.pipeline, self.vertex_buffer, self.index_buffer)
        {
            let start = self.start;
            let prepaint = move |bounds: Bounds<_>, window: &mut Window, _cx: &mut App| {
                let draw_bounds = inset_bounds(bounds, px(1.0));
                let viewport = window.viewport_size();
                let elapsed_seconds = start.elapsed().as_secs_f32();
                let uniform = build_scene_uniform(draw_bounds, viewport, elapsed_seconds);

                CustomDrawParams {
                    bounds: draw_bounds,
                    pipeline,
                    vertex_buffers: vec![CustomVertexBuffer {
                        source: CustomBufferSource::Buffer(vertex_buffer),
                    }],
                    vertex_count: 8,
                    index_buffer: Some(CustomIndexBuffer {
                        source: CustomBufferSource::Buffer(index_buffer),
                        format: CustomIndexFormat::U16,
                    }),
                    index_count: INDEX_COUNT,
                    target: None,
                    instance_count: 1,
                    push_constants: None,
                    bindings: vec![CustomBindingValue::Uniform(CustomBufferSource::Inline(
                        uniform,
                    ))],
                }
            };

            let paint = move |_bounds: Bounds<_>,
                              params: CustomDrawParams,
                              window: &mut Window,
                              _cx: &mut App| {
                if let Err(error) = window.paint_custom(params) {
                    log::error!("custom draw paint failed: {error}");
                }
            };

            div()
                .w(px(460.))
                .h(px(460.))
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

fn inset_bounds(bounds: Bounds<gpui::Pixels>, inset: gpui::Pixels) -> Bounds<gpui::Pixels> {
    let width = (bounds.size.width - inset * 2.0).max(px(1.0));
    let height = (bounds.size.height - inset * 2.0).max(px(1.0));
    Bounds {
        origin: bounds.origin + gpui::Point::new(inset, inset),
        size: gpui::Size::new(width, height),
    }
}

fn cube_vertex_data() -> Arc<[u8]> {
    let vertices: [[f32; 6]; 8] = [
        [-1.0, -1.0, -1.0, 1.0, 0.2, 0.2],
        [1.0, -1.0, -1.0, 0.2, 1.0, 0.2],
        [1.0, 1.0, -1.0, 0.2, 0.5, 1.0],
        [-1.0, 1.0, -1.0, 1.0, 1.0, 0.2],
        [-1.0, -1.0, 1.0, 1.0, 0.2, 1.0],
        [1.0, -1.0, 1.0, 0.2, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.6, 0.2],
        [-1.0, 1.0, 1.0, 0.8, 0.8, 0.8],
    ];

    let mut data = Vec::with_capacity(vertices.len() * 24);
    for vertex in vertices {
        for value in vertex {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    Arc::from(data)
}

fn cube_index_data() -> Arc<[u8]> {
    let indices: [u16; INDEX_COUNT as usize] = [
        0, 3, 2, 2, 1, 0, // back
        4, 5, 6, 6, 7, 4, // front
        0, 4, 7, 7, 3, 0, // left
        1, 2, 6, 6, 5, 1, // right
        3, 7, 6, 6, 2, 3, // top
        0, 1, 5, 5, 4, 0, // bottom
    ];

    let mut data = Vec::with_capacity(indices.len() * 2);
    for index in indices {
        data.extend_from_slice(&index.to_le_bytes());
    }
    Arc::from(data)
}

fn build_scene_uniform(
    bounds: Bounds<gpui::Pixels>,
    viewport: gpui::Size<gpui::Pixels>,
    elapsed_seconds: f32,
) -> Arc<[u8]> {
    let model = mat4_mul(
        mat4_rotation_y(elapsed_seconds * 0.9),
        mat4_rotation_x(elapsed_seconds * 0.6),
    );
    let view = mat4_look_at([0.0, 0.0, 4.8], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let projection = mat4_perspective_rh_zo(50f32.to_radians(), 1.0, 0.1, 50.0);
    let mvp = mat4_mul(projection, mat4_mul(view, model));

    let mut builder = CustomUniformBuilder::new();
    builder
        .push_mat4(mvp)
        .push_vec4(
            f32::from(bounds.origin.x),
            f32::from(bounds.origin.y),
            f32::from(bounds.size.width),
            f32::from(bounds.size.height),
        )
        .push_vec4(
            f32::from(viewport.width).max(1.0),
            f32::from(viewport.height).max(1.0),
            0.0,
            0.0,
        );
    builder.finish()
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

fn mat4_perspective_rh_zo(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let focal_length = 1.0 / (fov_y_radians * 0.5).tan();
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
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        (near * far) / (near - far),
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
        let bounds = Bounds::centered(None, size(px(560.), px(560.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(WindowDepthExample::new),
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
