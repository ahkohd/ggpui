# Overflow fade masks

Overflow fade masks let clipped content fade near container edges.

## Backend support matrix

| Backend | Status | Notes |
| --- | --- | --- |
| macOS (`gpui_macos`, Metal renderer) | Implemented | Edge fade masks on content masks |
| `gpui_wgpu` renderer backends | Implemented | Edge fade masks on content masks |
| Windows (`gpui_windows`, DirectX renderer) | Implemented | Edge fade masks on content masks |

## Example

Run from repository root:

```sh
cargo run -p gpui --example overflow_fade
```

## Styled API

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

## Behavior

- Fade is only applied on axes where overflow is clipped (`Hidden`, `Scroll`, or `Auto`).
- If an axis is `Visible`, fade on that axis is ignored.
- Fade distances are clamped to mask size.
- Opposite edge fades are normalized when their sum exceeds masked width/height.
- Nested content masks are intersected, and fade values are merged.

## Low-level API

`ContentMask` includes per-edge `fade_out`:

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
        // paint clipped & faded content
    },
);
```

## Notes and limitations

- Overflow masking is rectangular.
- `fade_out` is axis-aligned and evaluated in window space (not transform-aware).
