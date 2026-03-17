#![cfg_attr(target_family = "wasm", no_main)]

use gpui::{
    App, Bounds, Context, FontWeight, Render, Window, WindowBounds, WindowOptions, colors::Colors,
    div, prelude::*, px, size,
};
use gpui_platform::application;

struct OverflowFadeExample;

fn chat_row(index: usize, colors: &Colors) -> impl IntoElement {
    let background = if index.is_multiple_of(2) {
        colors.container
    } else {
        colors.background
    };

    div()
        .h_8()
        .px_3()
        .flex()
        .items_center()
        .rounded_md()
        .bg(background)
        .text_sm()
        .text_color(colors.text)
        .child(format!(
            "Row {:02}  Overflow fade mask demo item",
            index + 1
        ))
}

fn vertical_panel(title: &'static str, use_fade: bool, colors: &Colors) -> impl IntoElement {
    let scroll_area = div()
        .id(("vertical-scroll", if use_fade { 1_u32 } else { 0_u32 }))
        .h(px(320.0))
        .overflow_scroll()
        .rounded_lg()
        .bg(colors.background)
        .p_2()
        .child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .children((0..40).map(|index| chat_row(index, colors))),
        );
    let scroll_area = if use_fade {
        scroll_area.overflow_fade_y(px(22.0))
    } else {
        scroll_area
    };

    div()
        .w(px(320.0))
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::BOLD)
                .text_color(colors.text)
                .child(title),
        )
        .child(scroll_area)
        .child(
            div()
                .text_xs()
                .text_color(colors.disabled)
                .child(if use_fade {
                    "overflow_fade_y(px(22.0)) enabled"
                } else {
                    "No overflow_fade_y"
                }),
        )
}

fn horizontal_chip(index: usize, colors: &Colors) -> impl IntoElement {
    div()
        .h_8()
        .flex_none()
        .px_3()
        .rounded_full()
        .border_1()
        .border_color(colors.border)
        .bg(colors.container)
        .text_sm()
        .text_color(colors.text)
        .child(format!("Tag {:02}", index + 1))
}

fn horizontal_panel(colors: &Colors) -> impl IntoElement {
    div()
        .w(px(680.0))
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::BOLD)
                .text_color(colors.text)
                .child("Horizontal overflow fade"),
        )
        .child(
            div()
                .id("horizontal-scroll")
                .h(px(76.0))
                .overflow_x_scroll()
                .overflow_fade_x(px(30.0))
                .rounded_lg()
                .bg(colors.background)
                .child(
                    div()
                        .w(px(2200.0))
                        .flex()
                        .items_center()
                        .gap_2()
                        .px_2()
                        .h_full()
                        .children((0..50).map(|index| horizontal_chip(index, colors))),
                ),
        )
        .child(
            div()
                .text_xs()
                .text_color(colors.disabled)
                .child("overflow_fade_x(px(30.0)) enabled"),
        )
}

impl Render for OverflowFadeExample {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let colors = Colors::for_appearance(window);

        div()
            .id("app-scroll-root")
            .size_full()
            .p_6()
            .gap_6()
            .bg(colors.background)
            .overflow_scroll()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::BOLD)
                            .text_color(colors.text)
                            .child("Overflow Fade"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(colors.disabled)
                            .child("Compare no fade vs fade, then test horizontal fade."),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_4()
                    .items_start()
                    .child(vertical_panel("Without edge fade", false, &colors))
                    .child(vertical_panel("With edge fade", true, &colors)),
            )
            .child(horizontal_panel(&colors))
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(980.0), px(760.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| OverflowFadeExample),
        )
        .unwrap();
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
