# Custom Draw

Custom draw is implemented on the macOS Metal backend and on `gpui_wgpu` renderers, with a few documented backend-specific gaps on `gpui_wgpu`.

On backends without a custom draw registry, API entry points return an explicit error such as:

- `custom draw pipeline not supported on this platform`
- `custom compute pipeline not supported on this platform`

## Backend support matrix

| Backend | Status | Notes |
| --- | --- | --- |
| macOS (`gpui_macos`, Metal renderer) | Implemented | Supports window-target and offscreen custom render pipelines (multiple color targets, `Depth32Float`, MSAA for offscreen), custom compute pipelines, push constants, compressed texture uploads, explicit slots, binding arrays, profiling/diagnostics timing fields, and Metal-only pipeline inputs (`MSL` source / `.metallib` / pipeline cache path) |
| `gpui_wgpu` renderer backends | Implemented (with gaps) | Supports window-target custom render pipelines (single-sample color + optional `Depth32Float` depth), offscreen render pipelines (multiple color targets, `Depth32Float`, MSAA), custom compute pipelines with buffer/texture/sampler/uniform bindings, push constants (via WGSL rewrite to a generated uniform binding), buffer-backed texture uploads (including compressed formats with block-aligned rows), sampled `D2`/`D2Array`/`Cube` textures, 2D storage textures (`Rgba8Unorm`/`Bgra8Unorm`) in render/compute bindings, sampled compressed textures (BC/ETC2/ASTC when device features are available), explicit group/binding slots, binding arrays (buffer/texture/storage-texture arrays when required wgpu features are available), and per-frame profiling/diagnostics counters (including queue submit-to-complete latency, timestamp-query GPU time when supported, and derived submit/scheduled timing fields). Metal-only pipeline input APIs and PVRTC remain unsupported |

## Features

- Custom WGSL render pipelines
- Custom compute pipelines
- Vertex and index buffers
- Instanced rendering
- Uniform bindings and push constants
- Storage buffers with slices
- Storage textures and sampled textures
- Texture and sampler bindings
- Texture arrays and cubemaps
- Binding arrays for textures and buffers
- Block-compressed textures (BC, ETC2, ASTC, PVRTC)
- Offscreen render targets, depth testing, multiple color attachments, and MSAA
- Pipeline cache path for persistent Metal pipeline archives
- Pipeline creation from precompiled MSL or `.metallib`
- Per-frame GPU profile and frame diagnostics samples
- Resource diagnostics snapshot

## Examples

Run from `crates/gpui` package context:

```sh
cargo run -p gpui --example custom_draw_api
cargo run -p gpui --example custom_draw_api_animated
cargo run -p gpui --features bevy-ecs-example --example custom_draw_api_bevy_ecs
cargo run -p gpui --example custom_draw_api_rounded_clip
cargo run -p gpui --example custom_draw_api_instanced
cargo run -p gpui --example custom_draw_api_compute
cargo run -p gpui --example custom_draw_api_offscreen
cargo run -p gpui --example custom_draw_api_window_depth
cargo run -p gpui --example custom_draw_api_monkey
cargo run -p gpui --example custom_draw_api_gpu_profiling
cargo run -p gpui --example custom_draw_api_conformance
cargo run -p gpui --example custom_draw_api_mixed
cargo run -p gpui --example custom_draw_api_multi_group
cargo run -p gpui --example custom_draw_api_missing_binding
cargo run -p gpui --example custom_draw_api_binding_arrays
cargo run -p gpui --example custom_draw_api_texture_arrays
cargo run -p gpui --example custom_draw_api_cubemap
cargo run -p gpui --example custom_draw_api_storage_texture
cargo run -p gpui --example custom_draw_api_streaming_texture
cargo run -p gpui --example custom_draw_api_compressed_texture
# defaults to bundled bird_60fps.mp4 asset:
cargo run -p gpui --features video-ffmpeg --example custom_draw_api_video
# override with your own file:
cargo run -p gpui --features video-ffmpeg --example custom_draw_api_video -- --video /path/to/video.mp4
# force software decode path (disable hardware-accelerated decode):
cargo run -p gpui --features video-ffmpeg --example custom_draw_api_video -- --software
cargo run -p gpui --example custom_draw_api_metallib
cargo run -p gpui --example custom_draw_stress
```

The `custom_draw_api_video` window includes media controls (play/pause, restart, loop toggle), a seek scrubber, and current-time/total-duration display (including an in-video overlay).

Optional visual state-leak guard test (macOS):

```sh
cargo test -p gpui_platform --features "test-support,visual-test-guard" --test window_depth_state_leak_guard -- --nocapture
```

## Rounded corner clipping (opt-in)

Custom draw uses the current rectangular content mask for clipping. Parent corner radii are not
applied to canvas/custom-draw output automatically.

Use an opt-in shader helper when you need rounded clipping:

```wgsl
struct RoundedClipUniform {
  rect_size: vec2<f32>,
  corner_radius: f32,
  clip_enabled: f32,
};

fn rounded_rect_alpha(local_position: vec2<f32>, rect_size: vec2<f32>, corner_radius: f32) -> f32 {
  let half_size = rect_size * 0.5;
  let limited_corner_radius = clamp(corner_radius, 0.0, min(half_size.x, half_size.y));
  let centered_position = local_position - half_size;
  let q = abs(centered_position) - (half_size - vec2<f32>(limited_corner_radius, limited_corner_radius));
  let signed_distance = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - limited_corner_radius;
  let antialias_pixels = max(fwidth(signed_distance), 0.75);
  return 1.0 - smoothstep(0.0, antialias_pixels, signed_distance);
}
```

Example (side-by-side default vs helper):

```sh
cargo run -p gpui --example custom_draw_api_rounded_clip
```

## Runtime compressed-format selection

Use runtime capability queries before texture creation:

```rust
let format = if window.custom_texture_format_supported(CustomTextureFormat::Astc6x6Unorm)? {
    CustomTextureFormat::Astc6x6Unorm
} else if window.custom_texture_format_supported(CustomTextureFormat::Bc7Unorm)? {
    CustomTextureFormat::Bc7Unorm
} else {
    CustomTextureFormat::Rgba8Unorm
};
```

## Metal pipeline cache and precompiled libraries

```rust
window.set_custom_pipeline_cache_path("/tmp/gpui_custom_draw_pipeline_cache.binarchive")?;
```

Disable cache path:

```rust
window.clear_custom_pipeline_cache_path()?;
```

Create from MSL source:

```rust
let id = window.create_custom_pipeline_msl(desc, msl_source)?;
```

Create from `.metallib` file:

```rust
let id = window.create_custom_pipeline_metallib_file(desc, "path/to/custom.metallib")?;
```

## Notes and limitations

- Depth format support is currently `Depth32Float`.
- Window-target custom draws support single-sample depth testing (`Depth32Float`).
- Window-surface rendering uses one sample. MSAA is for offscreen targets.
- Canvas/custom-draw clipping uses rectangular content masks; rounded clipping is opt-in in shader code (see `custom_draw_api_rounded_clip`).
- Binding-array support through WGSL to MSL currently works for texture arrays.
- Buffer binding arrays in WGSL to MSL remain limited by translator support. Use precompiled MSL or `.metallib` when needed.
- GPU timestamp and frame diagnostics samples are sourced from Metal command buffer timing and callbacks.
- `gpui_wgpu` currently supports window-target (single-sample color + optional `Depth32Float`) and offscreen render pipelines (multiple color targets, `Depth32Float`, MSAA), custom compute pipelines with buffer/texture/sampler/uniform bindings, push constants (via WGSL rewrite to a generated uniform binding), explicit group/binding slots, binding arrays (buffer/texture/storage-texture arrays when required wgpu features are available), buffer-backed texture uploads (including compressed formats with block-aligned rows), sampled `D2`/`D2Array`/`Cube` textures, sampled compressed textures (BC/ETC2/ASTC when available), 2D storage textures (`Rgba8Unorm`/`Bgra8Unorm`) in render/compute bindings, and per-frame profiling/diagnostics counters (including queue submit-to-complete latency, timestamp-query GPU time when the adapter supports `TIMESTAMP_QUERY`, and derived submit/scheduled timing fields).
- Remaining non-parity items on `gpui_wgpu`: Metal-only pipeline input APIs (`create_pipeline_msl`, `create_pipeline_metallib(_file)`, pipeline cache path), PVRTC texture formats, and feature-dependent behavior when adapters do not expose required wgpu features.
- `custom_draw_api_video` requires enabling `video-ffmpeg` and installing FFmpeg development libraries/tooling (`pkg-config`/`vcpkg` plus `libavcodec`, `libavformat`, and `libswscale`).
- `custom_draw_api_bevy_ecs` requires enabling `bevy-ecs-example`.
