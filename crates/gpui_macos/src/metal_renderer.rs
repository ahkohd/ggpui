use crate::{
    custom_draw::{
        ArgumentBufferBinding, MetalBufferSnapshot, MetalCustomComputePipeline,
        MetalCustomDrawRegistry, MetalCustomPipeline,
    },
    metal_atlas::MetalAtlas,
};
use anyhow::{Context as _, Result, anyhow};
use block::ConcreteBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, CustomBindingKind, CustomBindingValue,
    CustomBufferSource, CustomDraw, CustomFrameDiagnostics, CustomGpuFrameProfile,
    CustomIndexBuffer, CustomIndexFormat, CustomTextureId, DevicePixels, PaintSurface, Path, Point,
    PrimitiveBatch, ScaledPixels, Scene, Size, point, size,
};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;

use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, MTLScissorRect,
    NSRange,
};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;

use std::{
    cell::Cell, collections::BTreeMap, ffi::c_void, mem, mem::MaybeUninit, ops::Range, ptr, slice,
    sync::Arc, time::Instant,
};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

pub(crate) type Context = Arc<Mutex<InstanceBufferPool>>;
pub(crate) type Renderer = MetalRenderer;

pub(crate) unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub(crate) struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub(crate) struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    depth_disabled_state: metal::DepthStencilState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    custom_draw: Arc<MetalCustomDrawRegistry>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    window_custom_depth_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests (captures screenshots without ScreenCaptureKit)
        #[cfg(any(test, feature = "test-support"))]
        layer.set_framebuffer_only(false);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }

        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        }
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        let is_unified_memory = device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let depth_disabled_state = create_depth_disabled_state(&device);

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let custom_draw = Arc::new(MetalCustomDrawRegistry::new(
            device.clone(),
            MTLPixelFormat::BGRA8Unorm,
        ));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            depth_disabled_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            custom_draw,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            window_custom_depth_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "test-support"))]
            headless_render_target: None,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn custom_draw_registry(&self) -> Arc<dyn gpui::CustomDrawRegistry> {
        self.custom_draw.clone()
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            let ns_size = NSSize {
                width: size.width.0 as f64,
                height: size.height.0 as f64,
            };
            unsafe {
                let _: () = msg_send![
                    layer.as_ref(),
                    setDrawableSize: ns_size
                ];
            }
        }
        self.update_path_intermediate_textures(size);
        self.update_window_custom_depth_texture(size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    fn update_window_custom_depth_texture(&mut self, size: Size<DevicePixels>) {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.window_custom_depth_texture = None;
            return;
        }

        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(size.width.0 as u64);
        descriptor.set_height(size.height.0 as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::Depth32Float);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        self.window_custom_depth_texture = Some(self.device.new_texture(&descriptor));
    }

    fn ensure_window_custom_depth_texture(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0.max(1) as u64;
        let height = size.height.0.max(1) as u64;
        let needs_resize = self
            .window_custom_depth_texture
            .as_ref()
            .is_none_or(|texture| texture.width() != width || texture.height() != height);

        if needs_resize {
            self.update_window_custom_depth_texture(size);
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let command_buffer = match self.render_frame(scene, drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let frame_encode_start = Instant::now();
        let custom_gpu_profile = self
            .custom_draw
            .gpu_profiling_enabled()
            .then(|| build_custom_gpu_profile(scene))
            .flatten();
        let custom_frame_diagnostics = self
            .custom_draw
            .frame_diagnostics_enabled()
            .then(|| build_custom_frame_diagnostics(scene))
            .flatten();

        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let instance_bindings = write_instances(scene, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
        )?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let block = ConcreteBlock::new(move |_| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
        });
        let block = block.copy();
        command_buffer.add_completed_handler(&block);
        register_custom_diagnostics(
            &command_buffer,
            self.custom_draw.clone(),
            custom_gpu_profile,
            custom_frame_diagnostics,
            frame_encode_start,
        );

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, drawable.texture(), viewport_size)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(drawable.texture())
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Update path intermediate textures for this size
        self.update_path_intermediate_textures(size);

        // Create an offscreen texture as render target
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Managed);
        let target_texture = self.device.new_texture(&texture_descriptor);

        let command_buffer = self.render_frame(scene, &target_texture, size)?;

        // On discrete GPUs (non-unified memory), Managed textures require an
        // explicit blit synchronize before the CPU can read back the rendered
        // data. Without this, get_bytes returns stale zeros.
        if !self.is_unified_memory {
            let blit = command_buffer.new_blit_command_encoder();
            blit.synchronize_resource(&target_texture);
            blit.end_encoding();
        }

        // Commit and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(&target_texture)
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        self.update_path_intermediate_textures(size);

        let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
            texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
        });
        if needs_new_target {
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
        }
        let target_texture = self
            .headless_render_target
            .clone()
            .expect("just ensured the render target exists");

        let command_buffer = self.render_frame(scene, &target_texture, size)?;

        // Commit without waiting, mirroring presentation to a real window where
        // the CPU doesn't block on the GPU.
        command_buffer.commit();
        Ok(())
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };

        self.ensure_window_custom_depth_texture(viewport_size);
        let window_custom_depth_texture = self.window_custom_depth_texture.clone();
        self.dispatch_custom_computes(scene, command_buffer, writer)?;
        self.draw_custom_render_targets(scene, command_buffer, writer)?;

        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            viewport_size,
            window_custom_depth_texture.as_deref(),
            metal::MTLLoadAction::Clear,
            Some(metal::MTLClearColor::new(0., 0., 0., alpha)),
        );

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        writer,
                        viewport_size,
                        command_buffer,
                    )?;

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        window_custom_depth_texture.as_deref(),
                        metal::MTLLoadAction::Load,
                        None,
                    );

                    if did_draw {
                        if let Err(error) = self.draw_paths_from_intermediate(
                            paths,
                            writer,
                            viewport_size,
                            command_encoder,
                        ) {
                            command_encoder.end_encoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    range.start,
                    instance_bindings,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Custom(range) => {
                    if let Err(error) = self.draw_custom_draws(
                        &scene.custom_draws[range],
                        writer,
                        command_encoder,
                        viewport_size,
                    ) {
                        command_encoder.end_encoding();
                        return Err(error);
                    }
                }
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.end_encoding();

        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
                content_mask: path.content_mask,
            }));
        }
        let vertex_instance_bindings = writer.write(&vertices)?;

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );

        command_encoder.end_encoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
            quads.start as u64,
        );
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        let sprite_instance_bindings = writer.write(&sprites)?;
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if surfaces.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    surface.image_buffer.get_width_of_plane(0),
                    surface.image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    surface.image_buffer.get_width_of_plane(1),
                    surface.image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            // let y_texture = y_texture.get_texture().unwrap().
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }
}

impl MetalRenderer {
    fn draw_custom_render_targets(
        &mut self,
        scene: &Scene,
        command_buffer: &metal::CommandBufferRef,
        writer: &mut InstanceBufferWriter,
    ) -> Result<()> {
        struct RenderTargetInfo {
            texture: metal::Texture,
            msaa_texture: Option<metal::Texture>,
            width: u32,
            height: u32,
            format: metal::MTLPixelFormat,
            clear_color: [f32; 4],
            is_render_target: bool,
            sample_count: u32,
        }

        struct DepthTargetInfo {
            texture: metal::Texture,
            format: metal::MTLPixelFormat,
            clear_depth: f64,
            width: u32,
            height: u32,
            sample_count: u32,
        }

        let mut draws_by_target = BTreeMap::new();
        for draw in scene.custom_draws.iter() {
            let Some(target) = draw.target.as_ref() else {
                continue;
            };
            let colors: Vec<u32> = target.colors.iter().map(|color| color.0).collect();
            draws_by_target
                .entry((colors, target.depth.map(|depth| depth.0)))
                .or_insert_with(Vec::new)
                .push(draw);
        }
        if draws_by_target.is_empty() {
            return Ok(());
        }

        let buffers_snapshot = self.custom_draw.buffers_snapshot();
        let textures_snapshot = self.custom_draw.textures_snapshot();
        let samplers_snapshot = self.custom_draw.samplers_snapshot();

        'render_target: for (_, draws) in draws_by_target {
            let Some(target) = draws.first().and_then(|draw| draw.target.as_ref()) else {
                continue;
            };
            let mut color_targets = Vec::with_capacity(target.colors.len());
            for color_id in &target.colors {
                let Some(color_target) =
                    self.custom_draw
                        .with_texture(*color_id, |entry| RenderTargetInfo {
                            texture: entry.texture.clone(),
                            msaa_texture: entry.msaa_texture.clone(),
                            width: entry.width,
                            height: entry.height,
                            format: entry.format,
                            clear_color: entry.clear_color,
                            is_render_target: entry.is_render_target,
                            sample_count: entry.sample_count,
                        })
                else {
                    log::warn!("custom render target {:?} missing", color_id.0);
                    continue 'render_target;
                };
                if !color_target.is_render_target {
                    log::warn!("custom draw target {:?} is not a render target", color_id.0);
                    continue 'render_target;
                }
                if color_target.sample_count > 1 && color_target.msaa_texture.is_none() {
                    log::warn!("custom draw target {:?} missing MSAA texture", color_id.0);
                    continue 'render_target;
                }
                color_targets.push(color_target);
            }
            let Some(first_target) = color_targets.first() else {
                continue;
            };
            for target_info in &color_targets[1..] {
                if target_info.width != first_target.width
                    || target_info.height != first_target.height
                {
                    log::warn!("custom render targets must match in size");
                    continue 'render_target;
                }
                if target_info.sample_count != first_target.sample_count {
                    log::warn!("custom render targets must match in sample count");
                    continue 'render_target;
                }
            }

            let depth_target = if let Some(depth_id) = target.depth {
                match self
                    .custom_draw
                    .with_depth_target(depth_id, |entry| DepthTargetInfo {
                        texture: entry.texture.clone(),
                        format: entry.format,
                        clear_depth: entry.clear_depth,
                        width: entry.width,
                        height: entry.height,
                        sample_count: entry.sample_count,
                    }) {
                    Some(target) => Some(target),
                    None => {
                        log::warn!("custom depth target {:?} missing", depth_id.0);
                        continue 'render_target;
                    }
                }
            } else {
                None
            };

            if let Some(depth_target) = depth_target.as_ref() {
                if depth_target.width != first_target.width
                    || depth_target.height != first_target.height
                {
                    log::warn!("custom depth target size mismatch");
                    continue 'render_target;
                }
                if depth_target.sample_count != first_target.sample_count {
                    log::warn!("custom depth target sample count mismatch");
                    continue 'render_target;
                }
            }

            let render_pass_descriptor = metal::RenderPassDescriptor::new();
            let color_attachments = render_pass_descriptor.color_attachments();
            for (index, color_target) in color_targets.iter().enumerate() {
                let Some(color_attachment) = color_attachments.object_at(index as u64) else {
                    log::warn!("custom draw color attachment {} missing", index);
                    continue 'render_target;
                };
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                if let Some(msaa_texture) = color_target.msaa_texture.as_ref() {
                    color_attachment.set_texture(Some(msaa_texture));
                    color_attachment.set_resolve_texture(Some(&color_target.texture));
                    color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
                } else {
                    color_attachment.set_texture(Some(&color_target.texture));
                    color_attachment.set_store_action(metal::MTLStoreAction::Store);
                }
                color_attachment.set_clear_color(metal::MTLClearColor::new(
                    color_target.clear_color[0] as f64,
                    color_target.clear_color[1] as f64,
                    color_target.clear_color[2] as f64,
                    color_target.clear_color[3] as f64,
                ));
            }

            if let Some(depth_target) = depth_target.as_ref() {
                let Some(depth_attachment) = render_pass_descriptor.depth_attachment() else {
                    log::warn!("custom draw depth attachment missing");
                    continue 'render_target;
                };
                depth_attachment.set_texture(Some(&depth_target.texture));
                depth_attachment.set_load_action(metal::MTLLoadAction::Clear);
                depth_attachment.set_store_action(metal::MTLStoreAction::Store);
                depth_attachment.set_clear_depth(depth_target.clear_depth);
            }

            let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
            command_encoder.set_viewport(metal::MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: first_target.width as f64,
                height: first_target.height as f64,
                znear: 0.0,
                zfar: 1.0,
            });

            let color_formats: Vec<metal::MTLPixelFormat> =
                color_targets.iter().map(|target| target.format).collect();
            let outcome = self.draw_custom_draws_for_target(
                &draws,
                writer,
                command_encoder,
                color_formats.as_slice(),
                first_target.sample_count,
                depth_target.as_ref().map(|target| target.format),
                None,
                &buffers_snapshot,
                &textures_snapshot,
                &samplers_snapshot,
            );
            command_encoder.end_encoding();

            if matches!(outcome, CustomDrawBindOutcome::OutOfSpace) {
                return Err(anyhow!("custom draw out of space"));
            }
        }

        Ok(())
    }

    fn dispatch_custom_computes(
        &mut self,
        scene: &Scene,
        command_buffer: &metal::CommandBufferRef,
        writer: &mut InstanceBufferWriter,
    ) -> Result<()> {
        if scene.custom_computes.is_empty() {
            return Ok(());
        }

        let buffers_snapshot = self.custom_draw.buffers_snapshot();
        let textures_snapshot = self.custom_draw.textures_snapshot();
        let samplers_snapshot = self.custom_draw.samplers_snapshot();

        let command_encoder = command_buffer.new_compute_command_encoder();
        for compute in scene.custom_computes.iter() {
            if compute.workgroup_count.contains(&0) {
                continue;
            }
            let Some(outcome) =
                self.custom_draw
                    .with_compute_pipeline(compute.pipeline, |pipeline| {
                        command_encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
                        match self.bind_custom_compute_resources(
                            command_encoder,
                            pipeline,
                            &compute.bindings,
                            &buffers_snapshot,
                            &textures_snapshot,
                            &samplers_snapshot,
                            writer,
                        ) {
                            CustomDrawBindOutcome::Ready => {}
                            other => return other,
                        }

                        let groups = metal::MTLSize {
                            width: compute.workgroup_count[0] as u64,
                            height: compute.workgroup_count[1] as u64,
                            depth: compute.workgroup_count[2] as u64,
                        };
                        let threads_per_group = metal::MTLSize {
                            width: pipeline.workgroup_size[0] as u64,
                            height: pipeline.workgroup_size[1] as u64,
                            depth: pipeline.workgroup_size[2] as u64,
                        };
                        command_encoder.dispatch_thread_groups(groups, threads_per_group);
                        CustomDrawBindOutcome::Ready
                    })
            else {
                log::warn!("custom compute pipeline {:?} not found", compute.pipeline.0);
                continue;
            };

            if matches!(outcome, CustomDrawBindOutcome::OutOfSpace) {
                command_encoder.end_encoding();
                return Err(anyhow!("custom compute out of space"));
            }
        }
        command_encoder.end_encoding();
        Ok(())
    }

    fn draw_custom_draws(
        &mut self,
        draws: &[CustomDraw],
        writer: &mut InstanceBufferWriter,
        command_encoder: &metal::RenderCommandEncoderRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<()> {
        let draws: Vec<&CustomDraw> = draws.iter().filter(|draw| draw.target.is_none()).collect();
        if draws.is_empty() {
            return Ok(());
        }

        let buffers_snapshot = self.custom_draw.buffers_snapshot();
        let textures_snapshot = self.custom_draw.textures_snapshot();
        let samplers_snapshot = self.custom_draw.samplers_snapshot();

        let color_formats = [self.custom_draw.surface_format()];
        let window_depth_format = self
            .window_custom_depth_texture
            .as_ref()
            .map(|_| metal::MTLPixelFormat::Depth32Float);
        let outcome = self.draw_custom_draws_for_target(
            &draws,
            writer,
            command_encoder,
            color_formats.as_slice(),
            1,
            window_depth_format,
            Some((
                viewport_size.width.0.max(0) as u32,
                viewport_size.height.0.max(0) as u32,
            )),
            &buffers_snapshot,
            &textures_snapshot,
            &samplers_snapshot,
        );
        command_encoder.set_depth_stencil_state(&self.depth_disabled_state);
        command_encoder.set_cull_mode(metal::MTLCullMode::None);
        command_encoder.set_scissor_rect(MTLScissorRect {
            x: 0,
            y: 0,
            width: viewport_size.width.0.max(0) as u64,
            height: viewport_size.height.0.max(0) as u64,
        });

        if matches!(outcome, CustomDrawBindOutcome::OutOfSpace) {
            return Err(anyhow!(
                "custom draw instance data exceeds the buffer limit"
            ));
        }
        Ok(())
    }

    fn draw_custom_draws_for_target(
        &mut self,
        draws: &[&CustomDraw],
        writer: &mut InstanceBufferWriter,
        command_encoder: &metal::RenderCommandEncoderRef,
        color_formats: &[metal::MTLPixelFormat],
        sample_count: u32,
        depth_format: Option<metal::MTLPixelFormat>,
        viewport_size: Option<(u32, u32)>,
        buffers_snapshot: &[Option<MetalBufferSnapshot>],
        textures_snapshot: &[Option<metal::Texture>],
        samplers_snapshot: &[Option<metal::SamplerState>],
    ) -> CustomDrawBindOutcome {
        let mut index = 0;
        while index < draws.len() {
            let batch_key = draws[index].batch_key;
            let mut end = index + 1;
            while end < draws.len() && draws[end].batch_key == batch_key {
                end += 1;
            }

            let batch = &draws[index..end];
            let pipeline_id = batch[0].pipeline;
            let bindings = &batch[0].bindings;

            let Some(outcome) = self.custom_draw.with_pipeline(pipeline_id, |pipeline| {
                if pipeline.color_formats.len() != color_formats.len() {
                    log::warn!(
                        "custom draw pipeline {:?} expects {} color targets, got {}",
                        pipeline_id.0,
                        pipeline.color_formats.len(),
                        color_formats.len()
                    );
                    return CustomDrawBindOutcome::SkipBatch;
                }
                for (expected, actual) in pipeline.color_formats.iter().zip(color_formats.iter()) {
                    if *expected != *actual {
                        log::warn!(
                            "custom draw pipeline {:?} color format mismatch",
                            pipeline_id.0
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                if pipeline.sample_count != sample_count {
                    log::warn!(
                        "custom draw pipeline {:?} sample count mismatch",
                        pipeline_id.0
                    );
                    return CustomDrawBindOutcome::SkipBatch;
                }
                if let Some(pipeline_depth_format) = pipeline.depth_format {
                    if Some(pipeline_depth_format) != depth_format {
                        log::warn!(
                            "custom draw pipeline {:?} depth format mismatch",
                            pipeline_id.0
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                if let Some(depth_state) = pipeline.depth_state.as_ref() {
                    command_encoder.set_depth_stencil_state(depth_state);
                } else {
                    command_encoder.set_depth_stencil_state(&self.depth_disabled_state);
                }

                command_encoder.set_render_pipeline_state(&pipeline.pipeline_state);
                command_encoder.set_cull_mode(pipeline.cull_mode);
                command_encoder.set_front_facing_winding(pipeline.front_face);

                match self.bind_custom_resources(
                    command_encoder,
                    pipeline,
                    bindings,
                    buffers_snapshot,
                    textures_snapshot,
                    samplers_snapshot,
                    writer,
                ) {
                    CustomDrawBindOutcome::Ready => {}
                    other => return other,
                }

                for draw in batch {
                    if draw.instance_count == 0 {
                        continue;
                    }
                    if let Some((viewport_width, viewport_height)) = viewport_size {
                        let Some((scissor_x, scissor_y, scissor_width, scissor_height)) =
                            clip_bounds_to_viewport(
                                draw.content_mask.bounds,
                                viewport_width,
                                viewport_height,
                            )
                        else {
                            continue;
                        };
                        command_encoder.set_scissor_rect(MTLScissorRect {
                            x: scissor_x as u64,
                            y: scissor_y as u64,
                            width: scissor_width as u64,
                            height: scissor_height as u64,
                        });
                    }
                    if draw.vertex_buffers.len() < pipeline.vertex_fetch_count {
                        log::warn!(
                            "custom draw missing vertex buffers (expected {}, got {})",
                            pipeline.vertex_fetch_count,
                            draw.vertex_buffers.len()
                        );
                        continue;
                    }

                    let mut vertex_binding_outcome = CustomDrawBindOutcome::Ready;
                    for (buffer_index, buffer) in draw
                        .vertex_buffers
                        .iter()
                        .enumerate()
                        .take(pipeline.vertex_fetch_count)
                    {
                        match self.bind_vertex_buffer(
                            command_encoder,
                            buffer_index,
                            &buffer.source,
                            buffers_snapshot,
                            writer,
                        ) {
                            CustomDrawBindOutcome::Ready => {}
                            other => {
                                vertex_binding_outcome = other;
                                break;
                            }
                        }
                    }

                    match vertex_binding_outcome {
                        CustomDrawBindOutcome::Ready => {
                            if let Some(index_buffer) = &draw.index_buffer {
                                if draw.index_count == 0 {
                                    continue;
                                }
                                match self.bind_index_buffer(
                                    index_buffer,
                                    draw.index_count,
                                    buffers_snapshot,
                                    writer,
                                ) {
                                    IndexBufferBindOutcome::Ready(binding) => {
                                        command_encoder.draw_indexed_primitives_instanced(
                                            pipeline.primitive,
                                            draw.index_count as u64,
                                            metal_index_type(index_buffer.format),
                                            binding.buffer.as_ref(),
                                            binding.offset,
                                            draw.instance_count as u64,
                                        );
                                    }
                                    IndexBufferBindOutcome::SkipBatch => continue,
                                    IndexBufferBindOutcome::OutOfSpace => {
                                        return CustomDrawBindOutcome::OutOfSpace;
                                    }
                                }
                            } else {
                                if draw.vertex_count == 0 {
                                    continue;
                                }
                                command_encoder.draw_primitives_instanced(
                                    pipeline.primitive,
                                    0,
                                    draw.vertex_count as u64,
                                    draw.instance_count as u64,
                                );
                            }
                        }
                        CustomDrawBindOutcome::SkipBatch => continue,
                        CustomDrawBindOutcome::OutOfSpace => {
                            return CustomDrawBindOutcome::OutOfSpace;
                        }
                    }
                }

                CustomDrawBindOutcome::Ready
            }) else {
                log::warn!("custom draw pipeline {:?} not found", pipeline_id.0);
                index = end;
                continue;
            };

            if matches!(outcome, CustomDrawBindOutcome::OutOfSpace) {
                return CustomDrawBindOutcome::OutOfSpace;
            }

            index = end;
        }

        CustomDrawBindOutcome::Ready
    }

    fn prepare_argument_buffer(
        &self,
        argument_binding: &ArgumentBufferBinding,
        writer: &mut InstanceBufferWriter,
    ) -> std::result::Result<InstanceBinding, InlineAllocationError> {
        let encoded_length = argument_binding.encoder.encoded_length() as usize;
        let alignment = argument_binding.encoder.alignment() as usize;
        let binding = allocate_inline_storage(writer, encoded_length, alignment)?;
        argument_binding
            .encoder
            .set_argument_buffer(&binding.buffer, binding.offset as metal::NSUInteger);
        Ok(binding)
    }

    fn bind_render_buffer_array(
        &self,
        command_encoder: &metal::RenderCommandEncoderRef,
        argument_binding: &ArgumentBufferBinding,
        buffer_slot: u64,
        sources: &[CustomBufferSource],
        buffers: &[Option<MetalBufferSnapshot>],
        writer: &mut InstanceBufferWriter,
    ) -> CustomDrawBindOutcome {
        let argument_buffer = match self.prepare_argument_buffer(argument_binding, writer) {
            Ok(binding) => binding,
            Err(InlineAllocationError::EmptyData) => {
                log::warn!("custom draw binding array argument buffer is empty");
                return CustomDrawBindOutcome::SkipBatch;
            }
            Err(InlineAllocationError::OutOfSpace) => {
                return CustomDrawBindOutcome::OutOfSpace;
            }
        };

        for (array_index, source) in sources.iter().enumerate() {
            let array_index = array_index as metal::NSUInteger;
            match source {
                CustomBufferSource::Inline(data) => match allocate_inline_bytes(writer, data) {
                    Ok(binding) => {
                        argument_binding.encoder.set_buffer(
                            array_index,
                            &binding.buffer,
                            binding.offset as u64,
                        );
                    }
                    Err(InlineAllocationError::EmptyData) => {
                        log::warn!("custom draw inline buffer array element is empty");
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    Err(InlineAllocationError::OutOfSpace) => {
                        return CustomDrawBindOutcome::OutOfSpace;
                    }
                },
                CustomBufferSource::Buffer(id) => {
                    let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom draw buffer {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    argument_binding
                        .encoder
                        .set_buffer(array_index, &buffer.buffer, 0);
                }
                CustomBufferSource::BufferSlice { id, offset, size } => {
                    let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom draw buffer {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    if *size == 0 {
                        log::warn!("custom draw buffer slice is empty");
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    if offset.saturating_add(*size) > buffer.size {
                        log::warn!("custom draw buffer slice out of range");
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    argument_binding.encoder.set_buffer(
                        array_index,
                        &buffer.buffer,
                        *offset as metal::NSUInteger,
                    );
                }
            }
        }

        command_encoder.set_vertex_buffer(
            buffer_slot,
            Some(&argument_buffer.buffer),
            argument_buffer.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            buffer_slot,
            Some(&argument_buffer.buffer),
            argument_buffer.offset as u64,
        );

        CustomDrawBindOutcome::Ready
    }

    fn bind_render_texture_array(
        &self,
        command_encoder: &metal::RenderCommandEncoderRef,
        argument_binding: &ArgumentBufferBinding,
        buffer_slot: u64,
        ids: &[CustomTextureId],
        textures: &[Option<metal::Texture>],
        writer: &mut InstanceBufferWriter,
    ) -> CustomDrawBindOutcome {
        let argument_buffer = match self.prepare_argument_buffer(argument_binding, writer) {
            Ok(binding) => binding,
            Err(InlineAllocationError::EmptyData) => {
                log::warn!("custom draw binding array argument buffer is empty");
                return CustomDrawBindOutcome::SkipBatch;
            }
            Err(InlineAllocationError::OutOfSpace) => {
                return CustomDrawBindOutcome::OutOfSpace;
            }
        };

        for (array_index, id) in ids.iter().enumerate() {
            let Some(slot) = textures.get(id.0 as usize) else {
                log::warn!("custom draw texture {:?} missing", id.0);
                return CustomDrawBindOutcome::SkipBatch;
            };
            let Some(texture) = slot.as_ref() else {
                log::warn!("custom draw texture {:?} missing", id.0);
                return CustomDrawBindOutcome::SkipBatch;
            };
            argument_binding
                .encoder
                .set_texture(array_index as metal::NSUInteger, texture);
        }

        command_encoder.set_vertex_buffer(
            buffer_slot,
            Some(&argument_buffer.buffer),
            argument_buffer.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            buffer_slot,
            Some(&argument_buffer.buffer),
            argument_buffer.offset as u64,
        );

        CustomDrawBindOutcome::Ready
    }

    fn bind_compute_buffer_array(
        &self,
        command_encoder: &metal::ComputeCommandEncoderRef,
        argument_binding: &ArgumentBufferBinding,
        buffer_slot: u64,
        sources: &[CustomBufferSource],
        buffers: &[Option<MetalBufferSnapshot>],
        writer: &mut InstanceBufferWriter,
    ) -> CustomDrawBindOutcome {
        let argument_buffer = match self.prepare_argument_buffer(argument_binding, writer) {
            Ok(binding) => binding,
            Err(InlineAllocationError::EmptyData) => {
                log::warn!("custom compute binding array argument buffer is empty");
                return CustomDrawBindOutcome::SkipBatch;
            }
            Err(InlineAllocationError::OutOfSpace) => {
                return CustomDrawBindOutcome::OutOfSpace;
            }
        };

        for (array_index, source) in sources.iter().enumerate() {
            let array_index = array_index as metal::NSUInteger;
            match source {
                CustomBufferSource::Inline(data) => match allocate_inline_bytes(writer, data) {
                    Ok(binding) => {
                        argument_binding.encoder.set_buffer(
                            array_index,
                            &binding.buffer,
                            binding.offset as u64,
                        );
                    }
                    Err(InlineAllocationError::EmptyData) => {
                        log::warn!("custom compute inline buffer array element is empty");
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    Err(InlineAllocationError::OutOfSpace) => {
                        return CustomDrawBindOutcome::OutOfSpace;
                    }
                },
                CustomBufferSource::Buffer(id) => {
                    let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom compute buffer {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    argument_binding
                        .encoder
                        .set_buffer(array_index, &buffer.buffer, 0);
                }
                CustomBufferSource::BufferSlice { id, offset, size } => {
                    let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom compute buffer {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    if *size == 0 {
                        log::warn!("custom compute buffer slice is empty");
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    if offset.saturating_add(*size) > buffer.size {
                        log::warn!("custom compute buffer slice out of range");
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    argument_binding.encoder.set_buffer(
                        array_index,
                        &buffer.buffer,
                        *offset as metal::NSUInteger,
                    );
                }
            }
        }

        command_encoder.set_buffer(
            buffer_slot,
            Some(&argument_buffer.buffer),
            argument_buffer.offset as u64,
        );

        CustomDrawBindOutcome::Ready
    }

    fn bind_compute_texture_array(
        &self,
        command_encoder: &metal::ComputeCommandEncoderRef,
        argument_binding: &ArgumentBufferBinding,
        buffer_slot: u64,
        ids: &[CustomTextureId],
        textures: &[Option<metal::Texture>],
        writer: &mut InstanceBufferWriter,
    ) -> CustomDrawBindOutcome {
        let argument_buffer = match self.prepare_argument_buffer(argument_binding, writer) {
            Ok(binding) => binding,
            Err(InlineAllocationError::EmptyData) => {
                log::warn!("custom compute binding array argument buffer is empty");
                return CustomDrawBindOutcome::SkipBatch;
            }
            Err(InlineAllocationError::OutOfSpace) => {
                return CustomDrawBindOutcome::OutOfSpace;
            }
        };

        for (array_index, id) in ids.iter().enumerate() {
            let Some(slot) = textures.get(id.0 as usize) else {
                log::warn!("custom compute texture {:?} missing", id.0);
                return CustomDrawBindOutcome::SkipBatch;
            };
            let Some(texture) = slot.as_ref() else {
                log::warn!("custom compute texture {:?} missing", id.0);
                return CustomDrawBindOutcome::SkipBatch;
            };
            argument_binding
                .encoder
                .set_texture(array_index as metal::NSUInteger, texture);
        }

        command_encoder.set_buffer(
            buffer_slot,
            Some(&argument_buffer.buffer),
            argument_buffer.offset as u64,
        );

        CustomDrawBindOutcome::Ready
    }

    fn bind_custom_resources(
        &self,
        command_encoder: &metal::RenderCommandEncoderRef,
        pipeline: &MetalCustomPipeline,
        bindings: &[CustomBindingValue],
        buffers: &[Option<MetalBufferSnapshot>],
        textures: &[Option<metal::Texture>],
        samplers: &[Option<metal::SamplerState>],
        writer: &mut InstanceBufferWriter,
    ) -> CustomDrawBindOutcome {
        if bindings.len() < pipeline.bindings.len() {
            log::warn!(
                "custom draw bindings missing (expected {}, got {})",
                pipeline.bindings.len(),
                bindings.len()
            );
        }

        for (index, kind) in pipeline.bindings.iter().enumerate() {
            let Some(binding) = bindings.get(index) else {
                match kind {
                    CustomBindingKind::Texture | CustomBindingKind::StorageTexture => {
                        command_encoder.set_vertex_texture(index as u64, None);
                        command_encoder.set_fragment_texture(index as u64, None);
                    }
                    CustomBindingKind::Sampler => {
                        command_encoder.set_vertex_sampler_state(index as u64, None);
                        command_encoder.set_fragment_sampler_state(index as u64, None);
                    }
                    _ => {}
                }
                continue;
            };
            let binding_index = index as u64;
            match (kind, binding) {
                (CustomBindingKind::Buffer, CustomBindingValue::Buffer(source)) => {
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_buffer_source(
                        command_encoder,
                        buffer_slot,
                        source,
                        buffers,
                        writer,
                        None,
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (
                    CustomBindingKind::BufferArray { count },
                    CustomBindingValue::BufferArray(sources),
                ) => {
                    if sources.len() != *count as usize {
                        log::warn!(
                            "custom draw buffer array length mismatch (expected {}, got {})",
                            count,
                            sources.len()
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    let Some(argument_binding) = pipeline
                        .argument_buffers
                        .get(index)
                        .and_then(|entry| entry.as_ref())
                    else {
                        log::warn!(
                            "custom draw binding array encoder missing at slot {}",
                            index
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_render_buffer_array(
                        command_encoder,
                        argument_binding,
                        buffer_slot,
                        sources,
                        buffers,
                        writer,
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (
                    CustomBindingKind::TextureArray { count }
                    | CustomBindingKind::StorageTextureArray { count },
                    CustomBindingValue::TextureArray(ids),
                ) => {
                    if ids.len() != *count as usize {
                        log::warn!(
                            "custom draw texture array length mismatch (expected {}, got {})",
                            count,
                            ids.len()
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    let Some(argument_binding) = pipeline
                        .argument_buffers
                        .get(index)
                        .and_then(|entry| entry.as_ref())
                    else {
                        log::warn!(
                            "custom draw binding array encoder missing at slot {}",
                            index
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_render_texture_array(
                        command_encoder,
                        argument_binding,
                        buffer_slot,
                        ids,
                        textures,
                        writer,
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (CustomBindingKind::Uniform { size }, CustomBindingValue::Uniform(source)) => {
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_buffer_source(
                        command_encoder,
                        buffer_slot,
                        source,
                        buffers,
                        writer,
                        Some(*size as usize),
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (
                    CustomBindingKind::Texture | CustomBindingKind::StorageTexture,
                    CustomBindingValue::Texture(id),
                ) => {
                    let Some(texture) = textures.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom draw texture {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    command_encoder.set_vertex_texture(binding_index, Some(texture));
                    command_encoder.set_fragment_texture(binding_index, Some(texture));
                }
                (CustomBindingKind::Sampler, CustomBindingValue::Sampler(id)) => {
                    let Some(sampler) = samplers.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom draw sampler {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    command_encoder.set_vertex_sampler_state(binding_index, Some(sampler));
                    command_encoder.set_fragment_sampler_state(binding_index, Some(sampler));
                }
                _ => {
                    log::warn!("custom draw binding mismatch at slot {}", index);
                    return CustomDrawBindOutcome::SkipBatch;
                }
            }
        }

        CustomDrawBindOutcome::Ready
    }

    fn bind_custom_compute_resources(
        &self,
        command_encoder: &metal::ComputeCommandEncoderRef,
        pipeline: &MetalCustomComputePipeline,
        bindings: &[CustomBindingValue],
        buffers: &[Option<MetalBufferSnapshot>],
        textures: &[Option<metal::Texture>],
        samplers: &[Option<metal::SamplerState>],
        writer: &mut InstanceBufferWriter,
    ) -> CustomDrawBindOutcome {
        if bindings.len() < pipeline.bindings.len() {
            log::warn!(
                "custom compute bindings missing (expected {}, got {})",
                pipeline.bindings.len(),
                bindings.len()
            );
        }

        for (index, kind) in pipeline.bindings.iter().enumerate() {
            let Some(binding) = bindings.get(index) else {
                match kind {
                    CustomBindingKind::Texture | CustomBindingKind::StorageTexture => {
                        command_encoder.set_texture(index as u64, None);
                    }
                    CustomBindingKind::Sampler => {
                        command_encoder.set_sampler_state(index as u64, None);
                    }
                    _ => {}
                }
                continue;
            };
            let binding_index = index as u64;
            match (kind, binding) {
                (CustomBindingKind::Buffer, CustomBindingValue::Buffer(source)) => {
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_compute_buffer_source(
                        command_encoder,
                        buffer_slot,
                        source,
                        buffers,
                        writer,
                        None,
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (
                    CustomBindingKind::BufferArray { count },
                    CustomBindingValue::BufferArray(sources),
                ) => {
                    if sources.len() != *count as usize {
                        log::warn!(
                            "custom compute buffer array length mismatch (expected {}, got {})",
                            count,
                            sources.len()
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    let Some(argument_binding) = pipeline
                        .argument_buffers
                        .get(index)
                        .and_then(|entry| entry.as_ref())
                    else {
                        log::warn!(
                            "custom compute binding array encoder missing at slot {}",
                            index
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_compute_buffer_array(
                        command_encoder,
                        argument_binding,
                        buffer_slot,
                        sources,
                        buffers,
                        writer,
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (
                    CustomBindingKind::TextureArray { count }
                    | CustomBindingKind::StorageTextureArray { count },
                    CustomBindingValue::TextureArray(ids),
                ) => {
                    if ids.len() != *count as usize {
                        log::warn!(
                            "custom compute texture array length mismatch (expected {}, got {})",
                            count,
                            ids.len()
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                    let Some(argument_binding) = pipeline
                        .argument_buffers
                        .get(index)
                        .and_then(|entry| entry.as_ref())
                    else {
                        log::warn!(
                            "custom compute binding array encoder missing at slot {}",
                            index
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_compute_texture_array(
                        command_encoder,
                        argument_binding,
                        buffer_slot,
                        ids,
                        textures,
                        writer,
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (CustomBindingKind::Uniform { size }, CustomBindingValue::Uniform(source)) => {
                    let buffer_slot = pipeline.buffer_binding_base + binding_index;
                    match self.bind_compute_buffer_source(
                        command_encoder,
                        buffer_slot,
                        source,
                        buffers,
                        writer,
                        Some(*size as usize),
                    ) {
                        CustomDrawBindOutcome::Ready => {}
                        other => return other,
                    }
                }
                (
                    CustomBindingKind::Texture | CustomBindingKind::StorageTexture,
                    CustomBindingValue::Texture(id),
                ) => {
                    let Some(texture) = textures.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom compute texture {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    command_encoder.set_texture(binding_index, Some(texture));
                }
                (CustomBindingKind::Sampler, CustomBindingValue::Sampler(id)) => {
                    let Some(sampler) = samplers.get(id.0 as usize).and_then(|slot| slot.as_ref())
                    else {
                        log::warn!("custom compute sampler {:?} missing", id.0);
                        return CustomDrawBindOutcome::SkipBatch;
                    };
                    command_encoder.set_sampler_state(binding_index, Some(sampler));
                }
                _ => {
                    log::warn!("custom compute binding mismatch at slot {}", index);
                    return CustomDrawBindOutcome::SkipBatch;
                }
            }
        }

        CustomDrawBindOutcome::Ready
    }

    fn bind_vertex_buffer(
        &self,
        command_encoder: &metal::RenderCommandEncoderRef,
        buffer_index: usize,
        source: &CustomBufferSource,
        buffers: &[Option<MetalBufferSnapshot>],
        writer: &mut InstanceBufferWriter,
    ) -> CustomDrawBindOutcome {
        match source {
            CustomBufferSource::Inline(data) => match allocate_inline_bytes(writer, data) {
                Ok(binding) => {
                    command_encoder.set_vertex_buffer(
                        buffer_index as u64,
                        Some(&binding.buffer),
                        binding.offset as u64,
                    );
                    CustomDrawBindOutcome::Ready
                }
                Err(InlineAllocationError::EmptyData) => {
                    log::warn!("custom draw inline vertex buffer is empty");
                    CustomDrawBindOutcome::SkipBatch
                }
                Err(InlineAllocationError::OutOfSpace) => CustomDrawBindOutcome::OutOfSpace,
            },
            CustomBufferSource::Buffer(id) => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom draw vertex buffer {:?} missing", id.0);
                    return CustomDrawBindOutcome::SkipBatch;
                };
                command_encoder.set_vertex_buffer(buffer_index as u64, Some(&buffer.buffer), 0);
                CustomDrawBindOutcome::Ready
            }
            CustomBufferSource::BufferSlice { id, offset, size } => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom draw vertex buffer {:?} missing", id.0);
                    return CustomDrawBindOutcome::SkipBatch;
                };
                if *size == 0 {
                    log::warn!("custom draw vertex buffer slice is empty");
                    return CustomDrawBindOutcome::SkipBatch;
                }
                if offset.saturating_add(*size) > buffer.size {
                    log::warn!("custom draw vertex buffer slice out of range");
                    return CustomDrawBindOutcome::SkipBatch;
                }
                command_encoder.set_vertex_buffer(
                    buffer_index as u64,
                    Some(&buffer.buffer),
                    *offset,
                );
                CustomDrawBindOutcome::Ready
            }
        }
    }

    fn bind_compute_buffer_source(
        &self,
        command_encoder: &metal::ComputeCommandEncoderRef,
        buffer_slot: u64,
        source: &CustomBufferSource,
        buffers: &[Option<MetalBufferSnapshot>],
        writer: &mut InstanceBufferWriter,
        expected_size: Option<usize>,
    ) -> CustomDrawBindOutcome {
        match source {
            CustomBufferSource::Inline(data) => {
                if let Some(expected_size) = expected_size {
                    if data.len() != expected_size {
                        log::warn!(
                            "custom compute uniform size mismatch (expected {}, got {})",
                            expected_size,
                            data.len()
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                match allocate_inline_bytes(writer, data) {
                    Ok(binding) => {
                        command_encoder.set_buffer(
                            buffer_slot,
                            Some(&binding.buffer),
                            binding.offset as u64,
                        );
                        CustomDrawBindOutcome::Ready
                    }
                    Err(InlineAllocationError::EmptyData) => {
                        log::warn!("custom compute inline buffer is empty");
                        CustomDrawBindOutcome::SkipBatch
                    }
                    Err(InlineAllocationError::OutOfSpace) => CustomDrawBindOutcome::OutOfSpace,
                }
            }
            CustomBufferSource::Buffer(id) => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom compute buffer {:?} missing", id.0);
                    return CustomDrawBindOutcome::SkipBatch;
                };
                if let Some(expected_size) = expected_size {
                    if buffer.size < expected_size as u64 {
                        log::warn!(
                            "custom compute uniform buffer too small (expected at least {}, got {})",
                            expected_size,
                            buffer.size
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                command_encoder.set_buffer(buffer_slot, Some(&buffer.buffer), 0);
                CustomDrawBindOutcome::Ready
            }
            CustomBufferSource::BufferSlice { id, offset, size } => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom compute buffer {:?} missing", id.0);
                    return CustomDrawBindOutcome::SkipBatch;
                };
                if *size == 0 {
                    log::warn!("custom compute buffer slice is empty");
                    return CustomDrawBindOutcome::SkipBatch;
                }
                if offset.saturating_add(*size) > buffer.size {
                    log::warn!("custom compute buffer slice out of range");
                    return CustomDrawBindOutcome::SkipBatch;
                }
                if let Some(expected_size) = expected_size {
                    if *size < expected_size as u64 {
                        log::warn!(
                            "custom compute uniform buffer slice too small (expected at least {}, got {})",
                            expected_size,
                            size
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                command_encoder.set_buffer(buffer_slot, Some(&buffer.buffer), *offset);
                CustomDrawBindOutcome::Ready
            }
        }
    }

    fn bind_index_buffer(
        &self,
        index_buffer: &CustomIndexBuffer,
        index_count: u32,
        buffers: &[Option<MetalBufferSnapshot>],
        writer: &mut InstanceBufferWriter,
    ) -> IndexBufferBindOutcome {
        let expected_len = index_count as usize * index_format_size(index_buffer.format);
        match &index_buffer.source {
            CustomBufferSource::Inline(data) => {
                if expected_len > 0 && data.len() < expected_len {
                    log::warn!(
                        "custom draw index buffer too small (expected at least {}, got {})",
                        expected_len,
                        data.len()
                    );
                    return IndexBufferBindOutcome::SkipBatch;
                }
                match allocate_inline_bytes(writer, data) {
                    Ok(binding) => IndexBufferBindOutcome::Ready(IndexBufferBinding {
                        buffer: binding.buffer,
                        offset: binding.offset as u64,
                    }),
                    Err(InlineAllocationError::EmptyData) => {
                        log::warn!("custom draw inline index buffer is empty");
                        IndexBufferBindOutcome::SkipBatch
                    }
                    Err(InlineAllocationError::OutOfSpace) => IndexBufferBindOutcome::OutOfSpace,
                }
            }
            CustomBufferSource::Buffer(id) => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom draw index buffer {:?} missing", id.0);
                    return IndexBufferBindOutcome::SkipBatch;
                };
                if expected_len > 0 && buffer.size < expected_len as u64 {
                    log::warn!(
                        "custom draw index buffer too small (expected at least {}, got {})",
                        expected_len,
                        buffer.size
                    );
                    return IndexBufferBindOutcome::SkipBatch;
                }
                IndexBufferBindOutcome::Ready(IndexBufferBinding {
                    buffer: buffer.buffer.clone(),
                    offset: 0,
                })
            }
            CustomBufferSource::BufferSlice { id, offset, size } => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom draw index buffer {:?} missing", id.0);
                    return IndexBufferBindOutcome::SkipBatch;
                };
                if *size == 0 {
                    log::warn!("custom draw index buffer slice is empty");
                    return IndexBufferBindOutcome::SkipBatch;
                }
                if offset.saturating_add(*size) > buffer.size {
                    log::warn!("custom draw index buffer slice out of range");
                    return IndexBufferBindOutcome::SkipBatch;
                }
                if expected_len > 0 && *size < expected_len as u64 {
                    log::warn!(
                        "custom draw index buffer slice too small (expected at least {}, got {})",
                        expected_len,
                        size
                    );
                    return IndexBufferBindOutcome::SkipBatch;
                }
                IndexBufferBindOutcome::Ready(IndexBufferBinding {
                    buffer: buffer.buffer.clone(),
                    offset: *offset,
                })
            }
        }
    }

    fn bind_buffer_source(
        &self,
        command_encoder: &metal::RenderCommandEncoderRef,
        buffer_slot: u64,
        source: &CustomBufferSource,
        buffers: &[Option<MetalBufferSnapshot>],
        writer: &mut InstanceBufferWriter,
        expected_size: Option<usize>,
    ) -> CustomDrawBindOutcome {
        match source {
            CustomBufferSource::Inline(data) => {
                if let Some(expected_size) = expected_size {
                    if data.len() != expected_size {
                        log::warn!(
                            "custom draw uniform size mismatch (expected {}, got {})",
                            expected_size,
                            data.len()
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                match allocate_inline_bytes(writer, data) {
                    Ok(binding) => {
                        command_encoder.set_vertex_buffer(
                            buffer_slot,
                            Some(&binding.buffer),
                            binding.offset as u64,
                        );
                        command_encoder.set_fragment_buffer(
                            buffer_slot,
                            Some(&binding.buffer),
                            binding.offset as u64,
                        );
                        CustomDrawBindOutcome::Ready
                    }
                    Err(InlineAllocationError::EmptyData) => {
                        log::warn!("custom draw inline buffer is empty");
                        CustomDrawBindOutcome::SkipBatch
                    }
                    Err(InlineAllocationError::OutOfSpace) => CustomDrawBindOutcome::OutOfSpace,
                }
            }
            CustomBufferSource::Buffer(id) => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom draw buffer {:?} missing", id.0);
                    return CustomDrawBindOutcome::SkipBatch;
                };
                if let Some(expected_size) = expected_size {
                    if buffer.size < expected_size as u64 {
                        log::warn!(
                            "custom draw uniform buffer too small (expected at least {}, got {})",
                            expected_size,
                            buffer.size
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                command_encoder.set_vertex_buffer(buffer_slot, Some(&buffer.buffer), 0);
                command_encoder.set_fragment_buffer(buffer_slot, Some(&buffer.buffer), 0);
                CustomDrawBindOutcome::Ready
            }
            CustomBufferSource::BufferSlice { id, offset, size } => {
                let Some(buffer) = buffers.get(id.0 as usize).and_then(|slot| slot.as_ref()) else {
                    log::warn!("custom draw buffer {:?} missing", id.0);
                    return CustomDrawBindOutcome::SkipBatch;
                };
                if *size == 0 {
                    log::warn!("custom draw buffer slice is empty");
                    return CustomDrawBindOutcome::SkipBatch;
                }
                if offset.saturating_add(*size) > buffer.size {
                    log::warn!("custom draw buffer slice out of range");
                    return CustomDrawBindOutcome::SkipBatch;
                }
                if let Some(expected_size) = expected_size {
                    if *size < expected_size as u64 {
                        log::warn!(
                            "custom draw uniform buffer slice too small (expected at least {}, got {})",
                            expected_size,
                            size
                        );
                        return CustomDrawBindOutcome::SkipBatch;
                    }
                }
                command_encoder.set_vertex_buffer(buffer_slot, Some(&buffer.buffer), *offset);
                command_encoder.set_fragment_buffer(buffer_slot, Some(&buffer.buffer), *offset);
                CustomDrawBindOutcome::Ready
            }
        }
    }
}

struct IndexBufferBinding {
    buffer: metal::Buffer,
    offset: u64,
}

enum IndexBufferBindOutcome {
    Ready(IndexBufferBinding),
    SkipBatch,
    OutOfSpace,
}

enum CustomDrawBindOutcome {
    Ready,
    SkipBatch,
    OutOfSpace,
}

enum InlineAllocationError {
    OutOfSpace,
    EmptyData,
}

fn allocate_inline_bytes(
    writer: &mut InstanceBufferWriter,
    data: &[u8],
) -> std::result::Result<InstanceBinding, InlineAllocationError> {
    if data.is_empty() {
        return Err(InlineAllocationError::EmptyData);
    }

    writer.write(data).map_err(|error| {
        log::error!("custom draw inline buffer allocation failed: {error:#}");
        InlineAllocationError::OutOfSpace
    })
}

fn allocate_inline_storage(
    writer: &mut InstanceBufferWriter,
    size: usize,
    alignment: usize,
) -> std::result::Result<InstanceBinding, InlineAllocationError> {
    if size == 0 {
        return Err(InlineAllocationError::EmptyData);
    }

    let (binding, storage) = writer
        .allocate_with_alignment::<u8>(size, alignment)
        .map_err(|error| {
            log::error!("custom draw argument buffer allocation failed: {error:#}");
            InlineAllocationError::OutOfSpace
        })?;
    for byte in storage {
        byte.write(0);
    }
    Ok(binding)
}

fn index_format_size(format: CustomIndexFormat) -> usize {
    match format {
        CustomIndexFormat::U16 => 2,
        CustomIndexFormat::U32 => 4,
    }
}

fn metal_index_type(format: CustomIndexFormat) -> metal::MTLIndexType {
    match format {
        CustomIndexFormat::U16 => metal::MTLIndexType::UInt16,
        CustomIndexFormat::U32 => metal::MTLIndexType::UInt32,
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    depth_texture: Option<&'a metal::TextureRef>,
    depth_load_action: metal::MTLLoadAction,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
    }

    if let Some(depth_texture) = depth_texture
        && let Some(depth_attachment) = render_pass_descriptor.depth_attachment()
    {
        depth_attachment.set_texture(Some(depth_texture));
        depth_attachment.set_load_action(depth_load_action);
        depth_attachment.set_store_action(metal::MTLStoreAction::Store);
        if matches!(depth_load_action, metal::MTLLoadAction::Clear) {
            depth_attachment.set_clear_depth(1.0);
        }
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

fn create_depth_disabled_state(device: &metal::DeviceRef) -> metal::DepthStencilState {
    let descriptor = metal::DepthStencilDescriptor::new();
    descriptor.set_depth_compare_function(metal::MTLCompareFunction::Always);
    descriptor.set_depth_write_enabled(false);
    device.new_depth_stencil_state(&descriptor)
}

#[cfg(any(test, feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_custom_work_counts(scene: &Scene) -> Option<(u32, u32, u32, u32)> {
    if scene.custom_draws.is_empty() && scene.custom_computes.is_empty() {
        return None;
    }

    let mut has_window_custom_draw = false;
    let mut offscreen_target_hashes = std::collections::HashSet::new();
    for draw in scene.custom_draws.iter() {
        if draw.target.is_some() {
            offscreen_target_hashes.insert(draw.batch_key.target_hash);
        } else {
            has_window_custom_draw = true;
        }
    }

    let custom_render_pass_count =
        offscreen_target_hashes.len() as u32 + u32::from(has_window_custom_draw);
    let custom_compute_pass_count = u32::from(!scene.custom_computes.is_empty());

    Some((
        scene.custom_draws.len() as u32,
        scene.custom_computes.len() as u32,
        custom_render_pass_count,
        custom_compute_pass_count,
    ))
}

fn build_custom_gpu_profile(scene: &Scene) -> Option<CustomGpuFrameProfile> {
    let (
        custom_draw_count,
        custom_compute_count,
        custom_render_pass_count,
        custom_compute_pass_count,
    ) = build_custom_work_counts(scene)?;

    Some(CustomGpuFrameProfile {
        custom_draw_count,
        custom_compute_count,
        custom_render_pass_count,
        custom_compute_pass_count,
        gpu_time_ns: None,
    })
}

fn build_custom_frame_diagnostics(scene: &Scene) -> Option<CustomFrameDiagnostics> {
    let (
        custom_draw_count,
        custom_compute_count,
        custom_render_pass_count,
        custom_compute_pass_count,
    ) = build_custom_work_counts(scene)?;

    Some(CustomFrameDiagnostics {
        custom_draw_count,
        custom_compute_count,
        custom_render_pass_count,
        custom_compute_pass_count,
        retry_count: 0,
        cpu_encode_time_ns: 0,
        submit_to_scheduled_ns: None,
        submit_to_completed_ns: None,
        scheduled_to_completed_ns: None,
        gpu_time_ns: None,
    })
}

fn metal_command_buffer_gpu_time_ns(command_buffer: &metal::CommandBufferRef) -> Option<u64> {
    #[allow(clippy::disallowed_methods)]
    unsafe {
        let has_gpu_start_time: bool =
            msg_send![command_buffer, respondsToSelector: sel!(GPUStartTime)];
        let has_gpu_end_time: bool =
            msg_send![command_buffer, respondsToSelector: sel!(GPUEndTime)];
        if !has_gpu_start_time || !has_gpu_end_time {
            return None;
        }

        let gpu_start_time: f64 = msg_send![command_buffer, GPUStartTime];
        let gpu_end_time: f64 = msg_send![command_buffer, GPUEndTime];
        if gpu_start_time <= 0.0 || gpu_end_time < gpu_start_time {
            return None;
        }

        let gpu_time_ns = ((gpu_end_time - gpu_start_time) * 1_000_000_000.0).round();
        if !gpu_time_ns.is_finite() || gpu_time_ns < 0.0 {
            return None;
        }

        Some(gpu_time_ns as u64)
    }
}

fn duration_as_u64_nanoseconds(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn register_custom_diagnostics(
    command_buffer: &metal::CommandBufferRef,
    custom_draw: Arc<MetalCustomDrawRegistry>,
    custom_gpu_profile: Option<CustomGpuFrameProfile>,
    custom_frame_diagnostics: Option<CustomFrameDiagnostics>,
    frame_encode_start: Instant,
) {
    if custom_gpu_profile.is_none() && custom_frame_diagnostics.is_none() {
        return;
    }

    let custom_frame_diagnostics = custom_frame_diagnostics.map(|mut diagnostics| {
        diagnostics.cpu_encode_time_ns = duration_as_u64_nanoseconds(frame_encode_start.elapsed());
        diagnostics
    });
    let submit_instant = Instant::now();
    let scheduled_instant = Arc::new(Mutex::new(None::<Instant>));

    if custom_frame_diagnostics.is_some() {
        let scheduled_instant = Arc::clone(&scheduled_instant);
        let scheduled_block = ConcreteBlock::new(move |_| {
            let mut scheduled_value = scheduled_instant.lock();
            if scheduled_value.is_none() {
                *scheduled_value = Some(Instant::now());
            }
        });
        let scheduled_block = scheduled_block.copy();
        command_buffer.add_scheduled_handler(&scheduled_block);
    }

    let completed_scheduled_instant = Arc::clone(&scheduled_instant);
    let completed_block = ConcreteBlock::new(move |completed_command_buffer| {
        let gpu_time_ns = metal_command_buffer_gpu_time_ns(completed_command_buffer);
        if let Some(mut custom_gpu_profile) = custom_gpu_profile {
            custom_gpu_profile.gpu_time_ns = gpu_time_ns;
            custom_draw.record_gpu_profile(custom_gpu_profile);
        }

        if let Some(mut diagnostics) = custom_frame_diagnostics {
            let completed_instant = Instant::now();
            let scheduled_instant = *completed_scheduled_instant.lock();
            diagnostics.gpu_time_ns = gpu_time_ns;
            diagnostics.submit_to_scheduled_ns = scheduled_instant.and_then(|scheduled| {
                scheduled
                    .checked_duration_since(submit_instant)
                    .map(duration_as_u64_nanoseconds)
            });
            diagnostics.submit_to_completed_ns = completed_instant
                .checked_duration_since(submit_instant)
                .map(duration_as_u64_nanoseconds);
            diagnostics.scheduled_to_completed_ns = scheduled_instant.and_then(|scheduled| {
                completed_instant
                    .checked_duration_since(scheduled)
                    .map(duration_as_u64_nanoseconds)
            });
            custom_draw.record_frame_diagnostics(diagnostics);
        }
    });
    let completed_block = completed_block.copy();
    command_buffer.add_completed_handler(&completed_block);
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: metal::Buffer,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    surfaces: InstanceBinding,
}

#[cfg_attr(not(test), allow(dead_code))]
fn clip_bounds_to_viewport(
    bounds: Bounds<ScaledPixels>,
    viewport_width: u32,
    viewport_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let min_x = bounds
        .origin
        .x
        .0
        .floor()
        .max(0.0)
        .min(viewport_width as f32) as u32;
    let min_y = bounds
        .origin
        .y
        .0
        .floor()
        .max(0.0)
        .min(viewport_height as f32) as u32;
    let max_x = (bounds.origin.x.0 + bounds.size.width.0)
        .ceil()
        .max(0.0)
        .min(viewport_width as f32) as u32;
    let max_y = (bounds.origin.y.0 + bounds.size.height.0)
        .ceil()
        .max(0.0)
        .min(viewport_height as f32) as u32;

    (max_x > min_x && max_y > min_y).then_some((min_x, min_y, max_x - min_x, max_y - min_y))
}

#[cfg(test)]
mod custom_draw_clip_tests {
    use super::*;
    use gpui::{point, size};

    #[test]
    fn custom_draw_clip_is_clamped_to_viewport() {
        let bounds = Bounds {
            origin: point(ScaledPixels(-2.2), ScaledPixels(3.2)),
            size: size(ScaledPixels(8.0), ScaledPixels(10.0)),
        };

        assert_eq!(clip_bounds_to_viewport(bounds, 5, 8), Some((0, 3, 5, 5)));
    }

    #[test]
    fn custom_draw_clip_rejects_bounds_outside_viewport() {
        let bounds = Bounds {
            origin: point(ScaledPixels(12.0), ScaledPixels(12.0)),
            size: size(ScaledPixels(4.0), ScaledPixels(4.0)),
        };

        assert_eq!(clip_bounds_to_viewport(bounds, 10, 10), None);
    }
}

fn write_instances(scene: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBindings> {
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
        }))?,
    })
}

struct InstanceBufferWriter {
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        self.allocate_with_alignment(count, INSTANCE_BUFFER_ALIGNMENT)
    }

    fn allocate_with_alignment<T>(
        &mut self,
        count: usize,
        alignment: usize,
    ) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let alignment = alignment.max(INSTANCE_BUFFER_ALIGNMENT);
        let mut offset = self.offset.next_multiple_of(alignment);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory);
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
            if offset > 0 {
                current.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: offset as NSUInteger,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[cfg(any(test, feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}
