//! Convenience crate that re-exports GPUI's platform traits and the
//! `current_platform` constructor so consumers don't need `#[cfg]` gating.

pub use gpui::Platform;

use std::rc::Rc;

/// Returns a background executor for the current platform.
pub fn background_executor() -> gpui::BackgroundExecutor {
    current_platform(true).background_executor()
}

pub fn application() -> gpui::Application {
    gpui::Application::with_platform(current_platform(false))
}

pub fn headless() -> gpui::Application {
    gpui::Application::with_platform(current_platform(true))
}

/// Unlike `application`, this function returns a single-threaded web application.
#[cfg(target_family = "wasm")]
pub fn single_threaded_web() -> gpui::Application {
    gpui::Application::with_platform(Rc::new(gpui_web::WebPlatform::new(false)))
}

/// Initializes panic hooks and logging for the web platform.
/// Call this before running the application in a wasm_bindgen entrypoint.
#[cfg(target_family = "wasm")]
pub fn web_init() {
    console_error_panic_hook::set_once();
    gpui_web::init_logging();
}

/// Returns the default [`Platform`] for the current OS.
pub fn current_platform(headless: bool) -> Rc<dyn Platform> {
    #[cfg(target_os = "macos")]
    {
        Rc::new(gpui_macos::MacPlatform::new(headless))
    }

    #[cfg(target_os = "windows")]
    {
        Rc::new(
            gpui_windows::WindowsPlatform::new(headless)
                .expect("failed to initialize Windows platform"),
        )
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    {
        gpui_linux::current_platform(headless)
    }

    #[cfg(target_family = "wasm")]
    {
        let _ = headless;
        Rc::new(gpui_web::WebPlatform::new(true))
    }
}

#[cfg(all(test, target_os = "macos", feature = "test-support"))]
mod tests {
    use super::*;
    use gpui::{
        App, AppContext, Bounds, Context, CustomBindingDesc, CustomBindingKind, CustomBindingName,
        CustomBindingValue, CustomBufferDesc, CustomBufferId, CustomBufferSource, CustomCullMode,
        CustomDepthCompare, CustomDepthFormat, CustomDepthState, CustomDrawParams,
        CustomIndexBuffer, CustomIndexFormat, CustomPipelineDesc, CustomPipelineId,
        CustomPipelineState, CustomPrimitiveTopology, CustomUniformBuilder, CustomVertexAttribute,
        CustomVertexAttributeName, CustomVertexBuffer, CustomVertexFetch, CustomVertexFormat,
        CustomVertexLayout, Empty, ParentElement, Pixels, Point, Render, Size, Styled,
        VisualTestAppContext, Window, canvas, div, px, size,
    };
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::time::Duration;

    // Note: All VisualTestAppContext tests are ignored by default because they require
    // the macOS main thread. Standard Rust tests run on worker threads, which causes
    // SIGABRT when interacting with macOS AppKit/Cocoa APIs.
    //
    // To run these tests, use:
    // cargo test -p gpui visual_test_context -- --ignored --test-threads=1

    #[test]
    #[ignore] // Requires macOS main thread
    fn test_foreground_tasks_run_with_run_until_parked() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        // Spawn a foreground task via the App's spawn method
        // This should use our TestDispatcher, not the MacDispatcher
        {
            let task_ran = task_ran.clone();
            cx.update(|cx| {
                cx.spawn(async move |_| {
                    *task_ran.borrow_mut() = true;
                })
                .detach();
            });
        }

        // The task should not have run yet
        assert!(!*task_ran.borrow());

        // Run until parked should execute the foreground task
        cx.run_until_parked();

        // Now the task should have run
        assert!(*task_ran.borrow());
    }

    #[test]
    #[ignore] // Requires macOS main thread
    fn test_advance_clock_triggers_delayed_tasks() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        // Spawn a task that waits for a timer
        {
            let task_ran = task_ran.clone();
            let executor = cx.background_executor.clone();
            cx.update(|cx| {
                cx.spawn(async move |_| {
                    executor.timer(Duration::from_millis(500)).await;
                    *task_ran.borrow_mut() = true;
                })
                .detach();
            });
        }

        // Run until parked - the task should be waiting on the timer
        cx.run_until_parked();
        assert!(!*task_ran.borrow());

        // Advance clock past the timer duration
        cx.advance_clock(Duration::from_millis(600));

        // Now the task should have completed
        assert!(*task_ran.borrow());
    }

    #[test]
    #[ignore] // Requires macOS main thread - window creation fails on test threads
    fn test_window_spawn_uses_test_dispatcher() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        let window = cx
            .open_offscreen_window_default(|_, cx| cx.new(|_| Empty))
            .expect("Failed to open window");

        // Spawn a task via window.spawn - this is the critical test case
        // for tooltip behavior, as tooltips use window.spawn for delayed show
        {
            let task_ran = task_ran.clone();
            cx.update_window(window.into(), |_, window, cx| {
                window
                    .spawn(cx, async move |_| {
                        *task_ran.borrow_mut() = true;
                    })
                    .detach();
            })
            .ok();
        }

        // The task should not have run yet
        assert!(!*task_ran.borrow());

        // Run until parked should execute the foreground task spawned via window
        cx.run_until_parked();

        // Now the task should have run
        assert!(*task_ran.borrow());
    }

    const STATE_LEAK_SHADER_SOURCE: &str = r#"
struct VertexInput {
  a0: vec3<f32>,
};

struct Uniforms {
  bounds: vec4<f32>,
  viewport: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
};

var<uniform> b0: Uniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let uv = input.a0.xy * 0.5 + vec2<f32>(0.5, 0.5);
  let pixel = b0.bounds.xy + uv * b0.bounds.zw;
  let ndc = vec2<f32>(
    (pixel.x / b0.viewport.x) * 2.0 - 1.0,
    1.0 - (pixel.y / b0.viewport.y) * 2.0
  );
  out.position = vec4<f32>(ndc, input.a0.z, 1.0);
  return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(0.95, 0.3, 0.3, 1.0);
}
"#;

    const STATE_LEAK_PANEL_WIDTH: f32 = 320.0;
    const STATE_LEAK_PANEL_HEIGHT: f32 = 320.0;

    struct WindowDepthStateLeakGuardView {
        pipeline: Option<CustomPipelineId>,
        vertex_buffer: Option<CustomBufferId>,
        index_buffer: Option<CustomBufferId>,
        error: Option<String>,
    }

    impl WindowDepthStateLeakGuardView {
        fn new(_cx: &mut Context<Self>) -> Self {
            Self {
                pipeline: None,
                vertex_buffer: None,
                index_buffer: None,
                error: None,
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
        ) -> gpui::Result<(CustomPipelineId, CustomBufferId, CustomBufferId)> {
            let pipeline = window.create_custom_pipeline(CustomPipelineDesc {
                name: "state_leak_guard_pipeline".to_string(),
                shader_source: STATE_LEAK_SHADER_SOURCE.to_string(),
                vertex_entry: "vs_main".to_string(),
                fragment_entry: "fs_main".to_string(),
                vertex_fetches: vec![CustomVertexFetch {
                    layout: CustomVertexLayout {
                        stride: 12,
                        attributes: vec![CustomVertexAttribute {
                            name: CustomVertexAttributeName::A0,
                            offset: 0,
                            format: CustomVertexFormat::F32Vec3,
                            location: None,
                        }],
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
                    kind: CustomBindingKind::Uniform { size: 32 },
                    slot: None,
                }],
            })?;

            let vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
                name: "state_leak_guard_vertices".to_string(),
                data: state_leak_vertex_data(),
            })?;

            let index_buffer = window.create_custom_buffer(CustomBufferDesc {
                name: "state_leak_guard_indices".to_string(),
                data: state_leak_index_data(),
            })?;

            Ok((pipeline, vertex_buffer, index_buffer))
        }
    }

    impl Render for WindowDepthStateLeakGuardView {
        fn render(
            &mut self,
            window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> impl gpui::IntoElement {
            self.ensure_resources(window);

            let panel = if let (Some(pipeline), Some(vertex_buffer), Some(index_buffer)) =
                (self.pipeline, self.vertex_buffer, self.index_buffer)
            {
                let prepaint = move |bounds: Bounds<_>, window: &mut Window, _cx: &mut App| {
                    let draw_bounds = inset_bounds(bounds, px(40.0));
                    let viewport_size = window.viewport_size();
                    let uniform = state_leak_uniform(draw_bounds, viewport_size);

                    CustomDrawParams {
                        bounds,
                        pipeline,
                        vertex_buffers: vec![CustomVertexBuffer {
                            source: CustomBufferSource::Buffer(vertex_buffer),
                        }],
                        vertex_count: 4,
                        index_buffer: Some(CustomIndexBuffer {
                            source: CustomBufferSource::Buffer(index_buffer),
                            format: CustomIndexFormat::U16,
                        }),
                        index_count: 6,
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
                        eprintln!("custom draw paint failed: {error}");
                    }
                };

                div()
                    .w(px(STATE_LEAK_PANEL_WIDTH))
                    .h(px(STATE_LEAK_PANEL_HEIGHT))
                    .border_1()
                    .border_color(gpui::white())
                    .bg(gpui::black())
                    .child(canvas(prepaint, paint).size_full())
            } else {
                div()
                    .w(px(STATE_LEAK_PANEL_WIDTH))
                    .h(px(STATE_LEAK_PANEL_HEIGHT))
                    .border_1()
                    .border_color(gpui::white())
                    .bg(gpui::black())
            };

            div()
                .size_full()
                .bg(gpui::black())
                .flex()
                .items_center()
                .justify_center()
                .child(panel)
        }
    }

    fn inset_bounds(bounds: Bounds<Pixels>, inset: Pixels) -> Bounds<Pixels> {
        let width = (bounds.size.width - inset * 2.0).max(px(1.0));
        let height = (bounds.size.height - inset * 2.0).max(px(1.0));
        Bounds {
            origin: bounds.origin + Point::new(inset, inset),
            size: Size::new(width, height),
        }
    }

    fn state_leak_vertex_data() -> Arc<[u8]> {
        let vertices: [[f32; 3]; 4] = [
            [-1.0, -1.0, 0.4],
            [1.0, -1.0, 0.4],
            [1.0, 1.0, 0.8],
            [-1.0, 1.0, 0.8],
        ];

        let mut data = Vec::with_capacity(vertices.len() * 12);
        for vertex in vertices {
            for component in vertex {
                data.extend_from_slice(&component.to_le_bytes());
            }
        }

        Arc::from(data)
    }

    fn state_leak_index_data() -> Arc<[u8]> {
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let mut data = Vec::with_capacity(indices.len() * 2);
        for index in indices {
            data.extend_from_slice(&index.to_le_bytes());
        }
        Arc::from(data)
    }

    fn state_leak_uniform(bounds: Bounds<Pixels>, viewport_size: Size<Pixels>) -> Arc<[u8]> {
        let mut builder = CustomUniformBuilder::new();
        builder
            .push_vec4(
                f32::from(bounds.origin.x),
                f32::from(bounds.origin.y),
                f32::from(bounds.size.width),
                f32::from(bounds.size.height),
            )
            .push_vec4(
                f32::from(viewport_size.width).max(1.0),
                f32::from(viewport_size.height).max(1.0),
                0.0,
                0.0,
            );
        builder.finish()
    }

    fn sample_luminance_at_logical_coordinate(
        screenshot: &image::RgbaImage,
        viewport_size: Size<Pixels>,
        x: f32,
        y: f32,
    ) -> u8 {
        let image_width = screenshot.width().max(1);
        let image_height = screenshot.height().max(1);
        let viewport_width = f32::from(viewport_size.width).max(1.0);
        let viewport_height = f32::from(viewport_size.height).max(1.0);

        let image_x = ((x / viewport_width) * image_width as f32)
            .round()
            .clamp(0.0, (image_width - 1) as f32) as u32;
        let image_y = ((y / viewport_height) * image_height as f32)
            .round()
            .clamp(0.0, (image_height - 1) as f32) as u32;

        let mut total_luminance = 0u32;
        let mut sample_count = 0u32;

        for offset_y in -1i32..=1 {
            for offset_x in -1i32..=1 {
                let sample_x = image_x
                    .saturating_add_signed(offset_x)
                    .min(image_width.saturating_sub(1));
                let sample_y = image_y
                    .saturating_add_signed(offset_y)
                    .min(image_height.saturating_sub(1));
                let pixel = screenshot.get_pixel(sample_x, sample_y);
                total_luminance = total_luminance
                    .saturating_add(u32::from(pixel[0]))
                    .saturating_add(u32::from(pixel[1]))
                    .saturating_add(u32::from(pixel[2]));
                sample_count = sample_count.saturating_add(3);
            }
        }

        if sample_count == 0 {
            return 0;
        }

        (total_luminance / sample_count).min(u32::from(u8::MAX)) as u8
    }

    #[test]
    #[ignore] // Requires macOS main thread
    fn test_window_custom_depth_draw_does_not_hide_panel_borders() {
        let mut cx = VisualTestAppContext::new(current_platform(false));
        let window = cx
            .open_offscreen_window(size(px(640.0), px(480.0)), |_, cx| {
                cx.new(WindowDepthStateLeakGuardView::new)
            })
            .expect("failed to open state leak guard window");

        cx.run_until_parked();

        let viewport_size = cx
            .update_window(window.into(), |_, window, _| window.viewport_size())
            .expect("failed to read viewport size");
        let screenshot = cx
            .capture_screenshot(window.into())
            .expect("failed to capture screenshot");

        let viewport_width = f32::from(viewport_size.width).max(1.0);
        let viewport_height = f32::from(viewport_size.height).max(1.0);
        let left = ((viewport_width - STATE_LEAK_PANEL_WIDTH) * 0.5).max(0.0);
        let top = ((viewport_height - STATE_LEAK_PANEL_HEIGHT) * 0.5).max(0.0);
        let right = left + STATE_LEAK_PANEL_WIDTH;
        let bottom = top + STATE_LEAK_PANEL_HEIGHT;
        let center_x = (left + right) * 0.5;
        let center_y = (top + bottom) * 0.5;

        let outside_luminance = sample_luminance_at_logical_coordinate(
            &screenshot,
            viewport_size,
            center_x,
            (top - 6.0).max(0.0),
        );
        let top_luminance =
            sample_luminance_at_logical_coordinate(&screenshot, viewport_size, center_x, top + 1.0);
        let bottom_luminance = sample_luminance_at_logical_coordinate(
            &screenshot,
            viewport_size,
            center_x,
            (bottom - 2.0).max(0.0),
        );
        let left_luminance = sample_luminance_at_logical_coordinate(
            &screenshot,
            viewport_size,
            left + 1.0,
            center_y,
        );
        let right_luminance = sample_luminance_at_logical_coordinate(
            &screenshot,
            viewport_size,
            (right - 2.0).max(0.0),
            center_y,
        );

        assert!(
            top_luminance > outside_luminance.saturating_add(20),
            "top border was not visible (outside={}, top={})",
            outside_luminance,
            top_luminance
        );
        assert!(
            bottom_luminance > outside_luminance.saturating_add(20),
            "bottom border was not visible (outside={}, bottom={})",
            outside_luminance,
            bottom_luminance
        );
        assert!(
            left_luminance > outside_luminance.saturating_add(20),
            "left border was not visible (outside={}, left={})",
            outside_luminance,
            left_luminance
        );
        assert!(
            right_luminance > outside_luminance.saturating_add(20),
            "right border was not visible (outside={}, right={})",
            outside_luminance,
            right_luminance
        );
    }
}
