# Custom Draw

Custom draw is implemented on the macOS Metal backend and has partial support on `gpui_wgpu` renderers.

On backends without a custom draw registry, API entry points return an explicit error such as:

- `custom draw pipeline not supported on this platform`
- `custom compute pipeline not supported on this platform`

## Backend support matrix

| Backend | Status | Notes |
| --- | --- | --- |
| macOS (`gpui_macos`, Metal renderer) | Implemented | Full custom draw and custom compute path in this fork |
| `gpui_wgpu` renderer backends | Partial | Supports window-target and offscreen custom render pipelines (multiple color targets, `Depth32Float`, MSAA), plus custom compute pipelines with buffer/texture/sampler/uniform bindings, buffer-backed texture uploads, sampled `D2`/`D2Array`/`Cube` textures, 2D storage textures (`Rgba8Unorm`/`Bgra8Unorm`), and sampled compressed textures (BC/ETC2/ASTC when device features are available). Explicit group/binding slots (including multi-group) work for non-array bindings. Binding arrays and push constants are not yet implemented |

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
cargo run -p gpui --example custom_draw_api_instanced
cargo run -p gpui --example custom_draw_api_compute
cargo run -p gpui --example custom_draw_api_offscreen
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
cargo run -p gpui --example custom_draw_api_metallib
cargo run -p gpui --example custom_draw_stress
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
- Window-surface rendering uses one sample. MSAA is for offscreen targets.
- Binding-array support through WGSL to MSL currently works for texture arrays.
- Buffer binding arrays in WGSL to MSL remain limited by translator support. Use precompiled MSL or `.metallib` when needed.
- GPU timestamp and frame diagnostics samples are sourced from Metal command buffer timing and callbacks.
- `gpui_wgpu` currently supports window-target and offscreen render pipelines (multiple color targets, `Depth32Float`, MSAA), custom compute pipelines with buffer/texture/sampler/uniform bindings, explicit group/binding slots for non-array bindings, buffer-backed texture uploads, sampled `D2`/`D2Array`/`Cube` textures, sampled compressed textures (BC/ETC2/ASTC when available), and 2D storage textures (`Rgba8Unorm`/`Bgra8Unorm`). PVRTC and other unsupported features still return explicit errors.
