# Superellipse corners

Superellipse corner rendering adds a smoothness control on top of existing rounded corners.

## Backend support matrix

| Backend | Status | Notes |
| --- | --- | --- |
| macOS (`gpui_macos`, Metal renderer) | Implemented | Superellipse rendering for quads/images/sprites |
| `gpui_wgpu` renderer backends | Implemented | Superellipse rendering for quads/images/sprites |
| Windows (`gpui_windows`, DirectX renderer) | Implemented | Superellipse rendering for quads/images/sprites |

## Example

Run from repository root:

```sh
cargo run -p gpui --example superellipse
```

## Styled API

Use `corner_superellipse(amount)` with values in `[0.0, 1.0]`:

- `0.0` → circular corner profile
- `0.5` → squircle-like profile
- `1.0` → square-ish profile

```rust
div()
    .rounded(px(24.0))
    .corner_superellipse(0.5)
```

Convenience methods are available:

- `.corner_superellipse_0()` ... `.corner_superellipse_1()`
- `.corner_superellipse_0p1()` ... `.corner_superellipse_0p9()`

## Low-level APIs

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

## Notes and limitations

- Smoothness is clamped to `[0.0, 1.0]`.
- Superellipse smoothing has no visible effect when corner radii are zero.
