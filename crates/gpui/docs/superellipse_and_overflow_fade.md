# Superellipse corners and overflow fade masks

This document covers two related visual features in GPUI:

- superellipse corner rendering (`corner_superellipse*`)
- edge fade masking for overflow content (`overflow_fade*`)

## Backend support matrix

| Backend | Status | Notes |
| --- | --- | --- |
| macOS (`gpui_macos`, Metal renderer) | Implemented | Superellipse rendering for quads/images/sprites and overflow fade masks |
| `gpui_wgpu` renderer backends | Implemented | Superellipse rendering for quads/images/sprites and overflow fade masks |
| Windows (`gpui_windows`, DirectX renderer) | Implemented | Superellipse rendering for quads/images/sprites and overflow fade masks |

## Examples

Run from `crates/gpui` package context:

```sh
cargo run -p gpui --example superellipse
cargo run -p gpui --example overflow_fade
```

## Superellipse corners

### Styled API

Use `corner_superellipse(amount)` with values in `[0.0, 1.0]`:

- `0.0` → circular corner profile
- `0.5` → squircle-like profile
- `1.0` → square-ish profile

```rust
div()
    .rounded(px(24.0))
    .corner_superellipse(0.5)
    .bg(rgb(0x8b5cf6))
```

Convenience methods are also available:

- `.corner_superellipse_0()` ... `.corner_superellipse_1()`
- `.corner_superellipse_0p1()` ... `.corner_superellipse_0p9()`

### Low-level painting API

```rust
window.paint_quad(
    quad(bounds, corner_radii, background, border_widths, border_color, border_style)
        .corner_superellipse(0.5),
);

window.paint_image_with_corner_superellipse(
    bounds,
    corner_radii,
    image,
    frame_index,
    grayscale,
    0.5,
)?;
```

## Overflow fade masks

### Styled API

Use per-edge or axis helpers:

```rust
div()
    .overflow_scroll()
    .overflow_fade_y(px(22.0))
```

```rust
div()
    .overflow_x_scroll()
    .overflow_fade_x(px(30.0))
```

```rust
div()
    .overflow_hidden()
    .overflow_fade(Edges::all(px(16.0).into()))
```

### Behavior details

- Fade is only applied on axes where overflow is clipped (`Hidden`, `Scroll`, or `Auto`).
- If an axis is `Visible`, fade on that axis is ignored.
- Fade distances are clamped to mask size.
- Opposite edge fades are normalized when their sum exceeds the masked width/height.
- Nested content masks are intersected, and fade values are merged accordingly.

### Low-level content mask API

`ContentMask` now includes per-edge `fade_out`:

```rust
window.with_content_mask(
    Some(ContentMask {
        bounds,
        fade_out: Edges {
            top: px(16.0),
            right: px(0.0),
            bottom: px(16.0),
            left: px(0.0),
        },
    }),
    |window| {
        // paint clipped + faded content
    },
);
```

## Notes and limitations

- Overflow masking is still rectangular.
- `fade_out` is axis-aligned and evaluated in window space (not transform-aware).
- Superellipse smoothing has no visible effect when corner radii are zero.
