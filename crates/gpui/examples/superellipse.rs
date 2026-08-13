#![cfg_attr(target_family = "wasm", no_main)]

use std::{fs, path::PathBuf};

use gpui::{
    App, AssetSource, Bounds, Context, FontWeight, ObjectFit, Render, SharedString, Window,
    WindowBounds, WindowOptions, div, img, prelude::*, px, rgb, size,
};
use gpui_platform::application;

const IMAGE_PATH: &str = "image/black-cat-typing.gif";

struct Assets {
    base: PathBuf,
}

impl AssetSource for Assets {
    fn load(&self, path: &str) -> anyhow::Result<Option<std::borrow::Cow<'static, [u8]>>> {
        fs::read(self.base.join(path))
            .map(|data| Some(std::borrow::Cow::Owned(data)))
            .map_err(Into::into)
    }

    fn list(&self, path: &str) -> anyhow::Result<Vec<SharedString>> {
        fs::read_dir(self.base.join(path))
            .map(|entries| {
                entries
                    .filter_map(|entry| {
                        entry
                            .ok()
                            .and_then(|entry| entry.file_name().into_string().ok())
                            .map(SharedString::from)
                    })
                    .collect()
            })
            .map_err(Into::into)
    }
}

struct SuperellipseDemo;

impl Render for SuperellipseDemo {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_4()
            .p_8()
            .bg(rgb(0xf0f0f0))
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::BOLD)
                    .child("Superellipse Corner Rendering"),
            )
            .child(
                div().flex().gap_4().children([
                    shape_card("Circular", 0.0, rgb(0x3b82f6)),
                    shape_card("Subtle", 0.3, rgb(0x06b6d4)),
                    shape_card("Squircle", 0.5, rgb(0x8b5cf6)),
                    shape_card("Rounded", 0.7, rgb(0xec4899)),
                ]),
            )
            .child(
                div()
                    .text_lg()
                    .font_weight(FontWeight::BOLD)
                    .child("Superellipse scale"),
            )
            .child(div().flex().gap_2().children((0..=10).map(|index| {
                let amount = index as f32 / 10.0;
                scale_chip(amount, rgb(0x10b981))
            })))
            .child(
                div()
                    .text_lg()
                    .font_weight(FontWeight::BOLD)
                    .child("Image with superellipse corners"),
            )
            .child(
                div().flex().gap_4().children([
                    image_card("Circular", 0.0),
                    image_card("Subtle", 0.3),
                    image_card("Squircle", 0.5),
                    image_card("Rounded", 0.7),
                ]),
            )
            .child(
                div()
                    .mt_4()
                    .p_4()
                    .rounded(px(8.0))
                    .bg(rgb(0x1f2937))
                    .text_sm()
                    .text_color(rgb(0xe5e7eb))
                    .font_family("monospace")
                    .child(
                        "div()\n    .rounded(px(40.0))\n    .corner_superellipse(0.5)\n\nimg(\"image/black-cat-typing.gif\")\n    .rounded(px(40.0))\n    .corner_superellipse(0.7)",
                    ),
            )
    }
}

fn shape_card(title: &'static str, amount: f32, color: gpui::Rgba) -> impl IntoElement {
    div().flex().flex_col().gap_2().items_center().children([
        div()
            .w(px(150.0))
            .h(px(150.0))
            .rounded(px(40.0))
            .corner_superellipse(amount)
            .bg(color)
            .shadow_lg(),
        div().font_weight(FontWeight::BOLD).child(title),
        div()
            .text_sm()
            .text_color(rgb(0x6b7280))
            .child(format!(".corner_superellipse({amount:.1})")),
    ])
}

fn scale_chip(amount: f32, color: gpui::Rgba) -> impl IntoElement {
    div()
        .w(px(60.0))
        .h(px(60.0))
        .rounded(px(15.0))
        .corner_superellipse(amount)
        .bg(color)
        .flex()
        .items_center()
        .justify_center()
        .text_xs()
        .text_color(gpui::white())
        .child(format!("{amount:.1}"))
}

fn image_card(title: &'static str, amount: f32) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .items_center()
        .child(
            img(IMAGE_PATH)
                .id(title)
                .w(px(150.0))
                .h(px(150.0))
                .rounded(px(40.0))
                .corner_superellipse(amount)
                .object_fit(ObjectFit::Cover),
        )
        .child(div().text_sm().text_color(rgb(0x6b7280)).child(title))
}

fn run_example() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");

    application()
        .with_assets(Assets { base })
        .run(|cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(960.0), px(760.0)), cx);

            if let Err(error) = cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| cx.new(|_| SuperellipseDemo),
            ) {
                log::error!("failed to open superellipse example window: {error}");
                cx.quit();
            }

            cx.activate(true);
        });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    env_logger::init();
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
