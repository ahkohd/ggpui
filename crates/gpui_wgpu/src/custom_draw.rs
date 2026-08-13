use anyhow::anyhow;
use gpui::{
    Bounds, CustomAddressMode, CustomBindingDesc, CustomBindingKind, CustomBindingSlot,
    CustomBindingValue, CustomBufferDesc, CustomBufferId, CustomBufferSource, CustomCompute,
    CustomComputePipelineDesc, CustomComputePipelineId, CustomCullMode, CustomDepthCompare,
    CustomDepthFormat, CustomDepthTargetDesc, CustomDepthTargetId, CustomDraw, CustomDrawRegistry,
    CustomDrawResourceStats, CustomFilterMode, CustomFrameDiagnostics, CustomFrontFace,
    CustomGpuFrameProfile, CustomIndexBuffer, CustomIndexFormat, CustomPipelineDesc,
    CustomPipelineId, CustomPrimitiveTopology, CustomPushConstantsDesc, CustomRenderTargetDesc,
    CustomSamplerDesc, CustomSamplerId, CustomTextureBufferUpdate, CustomTextureDesc,
    CustomTextureDimension, CustomTextureFormat, CustomTextureId, CustomTextureUpdate,
    CustomTextureUsage, CustomVertexFetch, CustomVertexFormat, Result, ScaledPixels,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::{NonZeroU32, NonZeroU64};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
struct WgpuBindingSpec {
    kind: CustomBindingKind,
    slot: CustomBindingSlot,
}

const MAX_SAMPLE_COUNT: u32 = 8;

#[derive(Clone)]
struct WgpuCustomPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    bindings: Vec<WgpuBindingSpec>,
    color_formats: Vec<wgpu::TextureFormat>,
    sample_count: u32,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_fetch_count: usize,
}

#[derive(Clone)]
struct WgpuCustomComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    bindings: Vec<WgpuBindingSpec>,
}

#[derive(Clone)]
struct WgpuCustomBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

#[derive(Clone)]
struct WgpuCustomTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    width: u32,
    height: u32,
    array_layer_count: u32,
    mip_level_count: u32,
    sample_count: u32,
    format: CustomTextureFormat,
    clear_color: [f32; 4],
    is_render_target: bool,
}

#[derive(Clone)]
struct WgpuCustomDepthTarget {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    sample_count: u32,
    clear_depth: f32,
    format: wgpu::TextureFormat,
}

#[derive(Default)]
struct WgpuCustomProfilingState {
    gpu_profiling_enabled: bool,
    frame_diagnostics_enabled: bool,
    last_gpu_profile: Option<CustomGpuFrameProfile>,
    last_frame_diagnostics: Option<CustomFrameDiagnostics>,
    last_submit_to_completed_ns: Option<u64>,
    last_gpu_time_ns: Option<u64>,
}

#[derive(Clone, Copy)]
struct BindingInfo {
    kind: CustomBindingKind,
    slot: CustomBindingSlot,
}

struct PushConstantsInfo {
    name: &'static str,
    size: u32,
    slot: CustomBindingSlot,
}

pub(crate) struct WgpuFrameGpuTimingCapture {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct WgpuStorageTextureBindingInfo {
    format: wgpu::TextureFormat,
    access: wgpu::StorageTextureAccess,
    view_dimension: wgpu::TextureViewDimension,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct WgpuSampledTextureBindingInfo {
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
    multisampled: bool,
}

struct OwnedBufferArrayElement {
    buffer: wgpu::Buffer,
    offset: u64,
    size: Option<NonZeroU64>,
}

enum OwnedBindingResource {
    Buffer {
        binding: u32,
        buffer: wgpu::Buffer,
        offset: u64,
        size: Option<NonZeroU64>,
    },
    BufferArray {
        binding: u32,
        elements: Vec<OwnedBufferArrayElement>,
    },
    Texture {
        binding: u32,
        view: wgpu::TextureView,
    },
    TextureArray {
        binding: u32,
        views: Vec<wgpu::TextureView>,
    },
    Sampler {
        binding: u32,
        sampler: wgpu::Sampler,
    },
}

pub(crate) struct WgpuCustomDrawRegistry {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    pipelines: Mutex<Vec<Option<WgpuCustomPipeline>>>,
    compute_pipelines: Mutex<Vec<Option<WgpuCustomComputePipeline>>>,
    buffers: Mutex<Vec<Option<WgpuCustomBuffer>>>,
    textures: Mutex<Vec<Option<WgpuCustomTexture>>>,
    depth_targets: Mutex<Vec<Option<WgpuCustomDepthTarget>>>,
    samplers: Mutex<Vec<Option<wgpu::Sampler>>>,
    profiling: Arc<Mutex<WgpuCustomProfilingState>>,
    timestamp_query_supported: bool,
}

impl WgpuCustomDrawRegistry {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let timestamp_query_supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        Self {
            device,
            queue,
            surface_format,
            pipelines: Mutex::new(Vec::new()),
            compute_pipelines: Mutex::new(Vec::new()),
            buffers: Mutex::new(Vec::new()),
            textures: Mutex::new(Vec::new()),
            depth_targets: Mutex::new(Vec::new()),
            samplers: Mutex::new(Vec::new()),
            profiling: Arc::new(Mutex::new(WgpuCustomProfilingState::default())),
            timestamp_query_supported,
        }
    }

    pub(crate) fn record_frame_metrics(
        &self,
        custom_draw_count: u32,
        custom_compute_count: u32,
        custom_render_pass_count: u32,
        custom_compute_pass_count: u32,
        retry_count: u32,
        cpu_encode_time_ns: u64,
    ) {
        let mut profiling = self.profiling.lock();
        let submit_to_completed_ns = profiling.last_submit_to_completed_ns.take();
        let gpu_time_ns = profiling.last_gpu_time_ns.take();
        let scheduled_to_completed_ns = gpu_time_ns;
        let submit_to_scheduled_ns = match (submit_to_completed_ns, scheduled_to_completed_ns) {
            (Some(submit_to_completed), Some(scheduled_to_completed))
                if submit_to_completed >= scheduled_to_completed =>
            {
                Some(submit_to_completed - scheduled_to_completed)
            }
            _ => None,
        };

        if profiling.gpu_profiling_enabled {
            profiling.last_gpu_profile = Some(CustomGpuFrameProfile {
                custom_draw_count,
                custom_compute_count,
                custom_render_pass_count,
                custom_compute_pass_count,
                gpu_time_ns,
            });
        }

        if profiling.frame_diagnostics_enabled {
            profiling.last_frame_diagnostics = Some(CustomFrameDiagnostics {
                custom_draw_count,
                custom_compute_count,
                custom_render_pass_count,
                custom_compute_pass_count,
                retry_count,
                cpu_encode_time_ns,
                submit_to_scheduled_ns,
                submit_to_completed_ns,
                scheduled_to_completed_ns,
                gpu_time_ns,
            });
        }
    }

    pub(crate) fn begin_frame_gpu_timing(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<WgpuFrameGpuTimingCapture> {
        if !self.timestamp_query_supported {
            return None;
        }

        let profiling = self.profiling.lock();
        let requested = profiling.gpu_profiling_enabled || profiling.frame_diagnostics_enabled;
        drop(profiling);
        if !requested {
            return None;
        }

        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("custom_draw_gpu_timing_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        let query_data_size = u64::from(wgpu::QUERY_SIZE) * 2;
        let resolve_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("custom_draw_gpu_timing_resolve"),
            size: query_data_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("custom_draw_gpu_timing_readback"),
            size: query_data_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.write_timestamp(&query_set, 0);

        Some(WgpuFrameGpuTimingCapture {
            query_set,
            resolve_buffer,
            readback_buffer,
        })
    }

    pub(crate) fn finish_frame_gpu_timing(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu_timing: WgpuFrameGpuTimingCapture,
    ) -> wgpu::Buffer {
        encoder.write_timestamp(&gpu_timing.query_set, 1);
        encoder.resolve_query_set(&gpu_timing.query_set, 0..2, &gpu_timing.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &gpu_timing.resolve_buffer,
            0,
            &gpu_timing.readback_buffer,
            0,
            u64::from(wgpu::QUERY_SIZE) * 2,
        );

        gpu_timing.readback_buffer
    }

    pub(crate) fn record_frame_gpu_timing(&self, readback_buffer: wgpu::Buffer) {
        let timestamp_period = self.queue.get_timestamp_period();
        let profiling = Arc::clone(&self.profiling);
        let callback_buffer = readback_buffer.clone();
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => {
                    let mapped = callback_buffer.slice(..).get_mapped_range();
                    if mapped.len() >= 16 {
                        let mut start_bytes = [0u8; 8];
                        start_bytes.copy_from_slice(&mapped[0..8]);
                        let mut end_bytes = [0u8; 8];
                        end_bytes.copy_from_slice(&mapped[8..16]);
                        let start = u64::from_le_bytes(start_bytes);
                        let end = u64::from_le_bytes(end_bytes);
                        if end >= start {
                            let ticks = end - start;
                            let gpu_time_ns =
                                ((ticks as f64) * f64::from(timestamp_period)).round() as u64;
                            profiling.lock().last_gpu_time_ns = Some(gpu_time_ns);
                        }
                    }
                    drop(mapped);
                    callback_buffer.unmap();
                }
                Err(error) => {
                    log::warn!("custom draw gpu timing readback failed: {error:?}");
                }
            });
    }

    pub(crate) fn record_submission_completion(&self) {
        let submit_start = Instant::now();
        let profiling = Arc::clone(&self.profiling);
        self.queue.on_submitted_work_done(move || {
            let elapsed_ns = submit_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            profiling.lock().last_submit_to_completed_ns = Some(elapsed_ns);
        });
    }

    pub(crate) fn draw_window_custom_draws(
        &self,
        draws: &[CustomDraw],
        encoder: &mut wgpu::CommandEncoder,
        frame_view: &wgpu::TextureView,
        window_depth_view: Option<&wgpu::TextureView>,
        window_depth_format: Option<wgpu::TextureFormat>,
        clear_window_depth: bool,
        viewport_width: u32,
        viewport_height: u32,
    ) -> u32 {
        if draws.is_empty() {
            return 0;
        }

        let window_draws: Vec<&CustomDraw> =
            draws.iter().filter(|draw| draw.target.is_none()).collect();
        if window_draws.is_empty() {
            return 0;
        }

        let pipelines = self.pipelines.lock().clone();
        let buffers = self.buffers.lock().clone();
        let textures = self.textures.lock().clone();
        let samplers = self.samplers.lock().clone();

        let depth_stencil_attachment =
            window_depth_view.map(|depth_view| wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if clear_window_depth {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("custom_draw_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment,
            ..Default::default()
        });

        let mut temporary_buffers: Vec<wgpu::Buffer> = Vec::new();
        let color_formats = [self.surface_format];
        self.draw_custom_draws_for_target(
            &window_draws,
            &pipelines,
            &buffers,
            &textures,
            &samplers,
            &mut pass,
            &color_formats,
            1,
            window_depth_format,
            Some((viewport_width, viewport_height)),
            &mut temporary_buffers,
        );

        1
    }

    pub(crate) fn draw_custom_render_targets(
        &self,
        draws: &[CustomDraw],
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if draws.is_empty() {
            return 0;
        }

        let mut draws_by_target: BTreeMap<(Vec<u32>, Option<u32>), Vec<&CustomDraw>> =
            BTreeMap::new();
        for draw in draws {
            let Some(target) = draw.target.as_ref() else {
                continue;
            };
            let colors: Vec<u32> = target.colors.iter().map(|color| color.0).collect();
            draws_by_target
                .entry((colors, target.depth.map(|depth| depth.0)))
                .or_default()
                .push(draw);
        }
        if draws_by_target.is_empty() {
            return 0;
        }

        let pipelines = self.pipelines.lock().clone();
        let buffers = self.buffers.lock().clone();
        let textures = self.textures.lock().clone();
        let depth_targets = self.depth_targets.lock().clone();
        let samplers = self.samplers.lock().clone();
        let mut render_pass_count = 0u32;

        'render_target: for (_, target_draws) in draws_by_target {
            let Some(target) = target_draws.first().and_then(|draw| draw.target.as_ref()) else {
                continue;
            };

            let mut color_targets = Vec::with_capacity(target.colors.len());
            for color_id in &target.colors {
                let Some(Some(color_target)) = textures.get(color_id.0 as usize) else {
                    log::warn!("custom render target {} is missing", color_id.0);
                    continue 'render_target;
                };
                if !color_target.is_render_target {
                    log::warn!("custom texture {} is not a render target", color_id.0);
                    continue 'render_target;
                }
                if color_target.sample_count > 1 && color_target.msaa_view.is_none() {
                    log::warn!("custom render target {} is missing MSAA data", color_id.0);
                    continue 'render_target;
                }
                color_targets.push(color_target.clone());
            }

            let Some(first_target) = color_targets.first() else {
                continue;
            };

            for color_target in &color_targets[1..] {
                if color_target.width != first_target.width
                    || color_target.height != first_target.height
                {
                    log::warn!("custom render targets must match in size");
                    continue 'render_target;
                }
                if color_target.sample_count != first_target.sample_count {
                    log::warn!("custom render targets must match in sample count");
                    continue 'render_target;
                }
            }

            let depth_target = if let Some(depth_id) = target.depth {
                let Some(Some(depth_target)) = depth_targets.get(depth_id.0 as usize) else {
                    log::warn!("custom depth target {} is missing", depth_id.0);
                    continue 'render_target;
                };
                Some(depth_target.clone())
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

            let mut color_formats = Vec::with_capacity(color_targets.len());
            for color_target in &color_targets {
                let Some(color_format) = map_texture_format(color_target.format) else {
                    log::warn!(
                        "custom render target format {:?} is not supported",
                        color_target.format
                    );
                    continue 'render_target;
                };
                color_formats.push(color_format);
            }

            let mut color_attachments = Vec::with_capacity(color_targets.len());
            for color_target in &color_targets {
                let attachment_view = color_target
                    .msaa_view
                    .as_ref()
                    .unwrap_or(&color_target.view);
                color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                    view: attachment_view,
                    resolve_target: color_target.msaa_view.as_ref().map(|_| &color_target.view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: color_target.clear_color[0] as f64,
                            g: color_target.clear_color[1] as f64,
                            b: color_target.clear_color[2] as f64,
                            a: color_target.clear_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }));
            }

            let depth_format = depth_target.as_ref().map(|target| target.format);
            let depth_stencil_attachment =
                depth_target
                    .as_ref()
                    .map(|depth_target| wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_target.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(depth_target.clear_depth),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    });

            render_pass_count = render_pass_count.saturating_add(1);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("custom_draw_offscreen_pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment,
                ..Default::default()
            });

            let mut temporary_buffers = Vec::new();
            self.draw_custom_draws_for_target(
                &target_draws,
                &pipelines,
                &buffers,
                &textures,
                &samplers,
                &mut pass,
                &color_formats,
                first_target.sample_count,
                depth_format,
                None,
                &mut temporary_buffers,
            );
        }

        render_pass_count
    }

    fn draw_custom_draws_for_target(
        &self,
        draws: &[&CustomDraw],
        pipelines: &[Option<WgpuCustomPipeline>],
        buffers: &[Option<WgpuCustomBuffer>],
        textures: &[Option<WgpuCustomTexture>],
        samplers: &[Option<wgpu::Sampler>],
        pass: &mut wgpu::RenderPass<'_>,
        color_formats: &[wgpu::TextureFormat],
        sample_count: u32,
        depth_format: Option<wgpu::TextureFormat>,
        viewport_size: Option<(u32, u32)>,
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) {
        for draw in draws {
            let pipeline_index = draw.pipeline.0 as usize;
            let Some(Some(pipeline)) = pipelines.get(pipeline_index) else {
                log::warn!("missing custom draw pipeline {}", draw.pipeline.0);
                continue;
            };

            if draw.vertex_buffers.len() != pipeline.vertex_fetch_count {
                log::warn!(
                    "custom draw pipeline {} expects {} vertex buffers, got {}",
                    draw.pipeline.0,
                    pipeline.vertex_fetch_count,
                    draw.vertex_buffers.len()
                );
                continue;
            }

            if draw.bindings.len() != pipeline.bindings.len() {
                log::warn!(
                    "custom draw pipeline {} expects {} bindings, got {}",
                    draw.pipeline.0,
                    pipeline.bindings.len(),
                    draw.bindings.len()
                );
                continue;
            }

            if !self.pipeline_matches_target(
                pipeline,
                draw.pipeline,
                color_formats,
                sample_count,
                depth_format,
            ) {
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
                pass.set_scissor_rect(scissor_x, scissor_y, scissor_width, scissor_height);
            }

            pass.set_pipeline(&pipeline.pipeline);

            let mut draw_failed = false;
            for (slot_index, vertex_buffer) in draw.vertex_buffers.iter().enumerate() {
                if let Err(error) = self.set_vertex_buffer(
                    pass,
                    slot_index as u32,
                    &vertex_buffer.source,
                    buffers,
                    temporary_buffers,
                ) {
                    log::warn!("custom draw vertex buffer binding failed: {error}");
                    draw_failed = true;
                    break;
                }
            }
            if draw_failed {
                continue;
            }

            let bind_groups = match self.create_draw_bind_groups(
                pipeline,
                &draw.bindings,
                buffers,
                textures,
                samplers,
                temporary_buffers,
            ) {
                Ok(bind_groups) => bind_groups,
                Err(error) => {
                    log::warn!("custom draw bind group creation failed: {error}");
                    continue;
                }
            };

            for (group, bind_group) in &bind_groups {
                pass.set_bind_group(*group, bind_group, &[]);
            }

            if let Some(index_buffer) = &draw.index_buffer {
                if let Err(error) =
                    self.set_index_buffer(pass, index_buffer, buffers, temporary_buffers)
                {
                    log::warn!("custom draw index buffer binding failed: {error}");
                    continue;
                }
                pass.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
            } else {
                pass.draw(0..draw.vertex_count, 0..draw.instance_count);
            }
        }
    }

    fn pipeline_matches_target(
        &self,
        pipeline: &WgpuCustomPipeline,
        pipeline_id: CustomPipelineId,
        color_formats: &[wgpu::TextureFormat],
        sample_count: u32,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> bool {
        if pipeline.color_formats.len() != color_formats.len() {
            log::warn!(
                "custom draw pipeline {} expects {} color targets, got {}",
                pipeline_id.0,
                pipeline.color_formats.len(),
                color_formats.len()
            );
            return false;
        }

        for (expected, actual) in pipeline.color_formats.iter().zip(color_formats.iter()) {
            if expected != actual {
                log::warn!(
                    "custom draw pipeline {} color target format mismatch",
                    pipeline_id.0
                );
                return false;
            }
        }

        if pipeline.sample_count != sample_count {
            log::warn!(
                "custom draw pipeline {} sample count mismatch (expected {}, got {})",
                pipeline_id.0,
                pipeline.sample_count,
                sample_count
            );
            return false;
        }

        if let Some(pipeline_depth_format) = pipeline.depth_format {
            if Some(pipeline_depth_format) != depth_format {
                log::warn!(
                    "custom draw pipeline {} depth format mismatch",
                    pipeline_id.0
                );
                return false;
            }
        }

        true
    }

    pub(crate) fn dispatch_custom_computes(
        &self,
        computes: &[CustomCompute],
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if computes.is_empty() {
            return 0;
        }

        let compute_pipelines = self.compute_pipelines.lock().clone();
        let buffers = self.buffers.lock().clone();
        let textures = self.textures.lock().clone();
        let samplers = self.samplers.lock().clone();

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("custom_compute_pass"),
            ..Default::default()
        });

        let mut temporary_buffers = Vec::new();

        for compute in computes {
            if compute.workgroup_count.contains(&0) {
                continue;
            }

            let pipeline_index = compute.pipeline.0 as usize;
            let Some(Some(pipeline)) = compute_pipelines.get(pipeline_index) else {
                log::warn!("missing custom compute pipeline {}", compute.pipeline.0);
                continue;
            };

            if compute.bindings.len() != pipeline.bindings.len() {
                log::warn!(
                    "custom compute pipeline {} expects {} bindings, got {}",
                    compute.pipeline.0,
                    pipeline.bindings.len(),
                    compute.bindings.len()
                );
                continue;
            }

            let bind_groups = match self.create_compute_bind_groups(
                pipeline,
                &compute.bindings,
                &buffers,
                &textures,
                &samplers,
                &mut temporary_buffers,
            ) {
                Ok(bind_groups) => bind_groups,
                Err(error) => {
                    log::warn!("custom compute bind group creation failed: {error}");
                    continue;
                }
            };

            pass.set_pipeline(&pipeline.pipeline);
            for (group, bind_group) in &bind_groups {
                pass.set_bind_group(*group, bind_group, &[]);
            }
            pass.dispatch_workgroups(
                compute.workgroup_count[0],
                compute.workgroup_count[1],
                compute.workgroup_count[2],
            );
        }

        1
    }

    fn set_vertex_buffer(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        slot_index: u32,
        source: &CustomBufferSource,
        buffers: &[Option<WgpuCustomBuffer>],
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<()> {
        match source {
            CustomBufferSource::Buffer(id) => {
                let Some(Some(buffer_entry)) = buffers.get(id.0 as usize) else {
                    return Err(anyhow!("custom draw buffer {} is missing", id.0));
                };
                pass.set_vertex_buffer(slot_index, buffer_entry.buffer.slice(..));
            }
            CustomBufferSource::BufferSlice { id, offset, size } => {
                let Some(Some(buffer_entry)) = buffers.get(id.0 as usize) else {
                    return Err(anyhow!("custom draw buffer {} is missing", id.0));
                };
                let end = offset
                    .checked_add(*size)
                    .ok_or_else(|| anyhow!("custom draw buffer slice overflow"))?;
                if end > buffer_entry.size {
                    return Err(anyhow!(
                        "custom draw vertex buffer slice out of bounds (offset {} size {} buffer {})",
                        offset,
                        size,
                        buffer_entry.size
                    ));
                }
                pass.set_vertex_buffer(slot_index, buffer_entry.buffer.slice(*offset..end));
            }
            CustomBufferSource::Inline(data) => {
                let inline_buffer =
                    self.create_inline_buffer(data, wgpu::BufferUsages::VERTEX, "custom_vertex");
                pass.set_vertex_buffer(slot_index, inline_buffer.slice(..));
                temporary_buffers.push(inline_buffer);
            }
        }

        Ok(())
    }

    fn set_index_buffer(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        index_buffer: &CustomIndexBuffer,
        buffers: &[Option<WgpuCustomBuffer>],
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<()> {
        let index_format = map_index_format(index_buffer.format);

        match &index_buffer.source {
            CustomBufferSource::Buffer(id) => {
                let Some(Some(buffer_entry)) = buffers.get(id.0 as usize) else {
                    return Err(anyhow!("custom draw buffer {} is missing", id.0));
                };
                pass.set_index_buffer(buffer_entry.buffer.slice(..), index_format);
            }
            CustomBufferSource::BufferSlice { id, offset, size } => {
                let Some(Some(buffer_entry)) = buffers.get(id.0 as usize) else {
                    return Err(anyhow!("custom draw buffer {} is missing", id.0));
                };
                let end = offset
                    .checked_add(*size)
                    .ok_or_else(|| anyhow!("custom draw buffer slice overflow"))?;
                if end > buffer_entry.size {
                    return Err(anyhow!(
                        "custom draw index buffer slice out of bounds (offset {} size {} buffer {})",
                        offset,
                        size,
                        buffer_entry.size
                    ));
                }
                pass.set_index_buffer(buffer_entry.buffer.slice(*offset..end), index_format);
            }
            CustomBufferSource::Inline(data) => {
                let inline_buffer =
                    self.create_inline_buffer(data, wgpu::BufferUsages::INDEX, "custom_index");
                pass.set_index_buffer(inline_buffer.slice(..), index_format);
                temporary_buffers.push(inline_buffer);
            }
        }

        Ok(())
    }

    fn create_draw_bind_groups(
        &self,
        pipeline: &WgpuCustomPipeline,
        binding_values: &[CustomBindingValue],
        buffers: &[Option<WgpuCustomBuffer>],
        textures: &[Option<WgpuCustomTexture>],
        samplers: &[Option<wgpu::Sampler>],
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<Vec<(u32, wgpu::BindGroup)>> {
        self.create_bind_groups(
            "custom_draw_bind_group",
            &pipeline.bind_group_layouts,
            &pipeline.bindings,
            binding_values,
            buffers,
            textures,
            samplers,
            temporary_buffers,
        )
    }

    fn create_compute_bind_groups(
        &self,
        pipeline: &WgpuCustomComputePipeline,
        binding_values: &[CustomBindingValue],
        buffers: &[Option<WgpuCustomBuffer>],
        textures: &[Option<WgpuCustomTexture>],
        samplers: &[Option<wgpu::Sampler>],
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<Vec<(u32, wgpu::BindGroup)>> {
        self.create_bind_groups(
            "custom_compute_bind_group",
            &pipeline.bind_group_layouts,
            &pipeline.bindings,
            binding_values,
            buffers,
            textures,
            samplers,
            temporary_buffers,
        )
    }

    fn create_bind_groups(
        &self,
        label: &str,
        bind_group_layouts: &[wgpu::BindGroupLayout],
        binding_specs: &[WgpuBindingSpec],
        binding_values: &[CustomBindingValue],
        buffers: &[Option<WgpuCustomBuffer>],
        textures: &[Option<WgpuCustomTexture>],
        samplers: &[Option<wgpu::Sampler>],
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<Vec<(u32, wgpu::BindGroup)>> {
        let mut owned_resources_by_group: std::collections::BTreeMap<
            u32,
            Vec<OwnedBindingResource>,
        > = std::collections::BTreeMap::new();

        for (binding_spec, binding_value) in binding_specs.iter().zip(binding_values.iter()) {
            let owned_resources = owned_resources_by_group
                .entry(binding_spec.slot.group)
                .or_default();

            match (binding_spec.kind, binding_value) {
                (CustomBindingKind::Buffer, CustomBindingValue::Buffer(source)) => {
                    let resource = self.resolve_buffer_resource(
                        binding_spec.slot.binding,
                        source,
                        buffers,
                        wgpu::BufferUsages::STORAGE,
                        temporary_buffers,
                    )?;
                    owned_resources.push(resource);
                }
                (
                    CustomBindingKind::BufferArray { count },
                    CustomBindingValue::BufferArray(sources),
                ) => {
                    if sources.len() != count as usize {
                        return Err(anyhow!(
                            "custom draw buffer binding array length mismatch (expected {}, got {})",
                            count,
                            sources.len()
                        ));
                    }

                    let mut elements = Vec::with_capacity(sources.len());
                    for source in sources {
                        let resource = self.resolve_buffer_resource(
                            binding_spec.slot.binding,
                            source,
                            buffers,
                            wgpu::BufferUsages::STORAGE,
                            temporary_buffers,
                        )?;
                        let OwnedBindingResource::Buffer {
                            buffer,
                            offset,
                            size,
                            ..
                        } = resource
                        else {
                            return Err(anyhow!("expected buffer resource for buffer array"));
                        };
                        elements.push(OwnedBufferArrayElement {
                            buffer,
                            offset,
                            size,
                        });
                    }

                    owned_resources.push(OwnedBindingResource::BufferArray {
                        binding: binding_spec.slot.binding,
                        elements,
                    });
                }
                (CustomBindingKind::Uniform { size }, CustomBindingValue::Uniform(source)) => {
                    let resource = self.resolve_buffer_resource(
                        binding_spec.slot.binding,
                        source,
                        buffers,
                        wgpu::BufferUsages::UNIFORM,
                        temporary_buffers,
                    )?;

                    let OwnedBindingResource::Buffer {
                        binding,
                        buffer,
                        offset,
                        size: available_size,
                    } = resource
                    else {
                        return Err(anyhow!("expected buffer resource for uniform binding"));
                    };

                    let Some(required_size) = NonZeroU64::new(size as u64) else {
                        return Err(anyhow!("uniform binding declared size must be non-zero"));
                    };

                    let available_size = available_size.map(|value| value.get()).unwrap_or(0);
                    if available_size < required_size.get() {
                        return Err(anyhow!(
                            "uniform binding is smaller than declared size (have {}, need {})",
                            available_size,
                            required_size
                        ));
                    }

                    owned_resources.push(OwnedBindingResource::Buffer {
                        binding,
                        buffer,
                        offset,
                        size: Some(required_size),
                    });
                }
                (CustomBindingKind::Texture, CustomBindingValue::Texture(id))
                | (CustomBindingKind::StorageTexture, CustomBindingValue::Texture(id)) => {
                    let Some(Some(texture_entry)) = textures.get(id.0 as usize) else {
                        return Err(anyhow!("custom draw texture {} is missing", id.0));
                    };
                    owned_resources.push(OwnedBindingResource::Texture {
                        binding: binding_spec.slot.binding,
                        view: texture_entry.view.clone(),
                    });
                }
                (
                    CustomBindingKind::TextureArray { count }
                    | CustomBindingKind::StorageTextureArray { count },
                    CustomBindingValue::TextureArray(ids),
                ) => {
                    if ids.len() != count as usize {
                        return Err(anyhow!(
                            "custom draw texture binding array length mismatch (expected {}, got {})",
                            count,
                            ids.len()
                        ));
                    }

                    let mut views = Vec::with_capacity(ids.len());
                    for id in ids {
                        let Some(Some(texture_entry)) = textures.get(id.0 as usize) else {
                            return Err(anyhow!("custom draw texture {} is missing", id.0));
                        };
                        views.push(texture_entry.view.clone());
                    }

                    owned_resources.push(OwnedBindingResource::TextureArray {
                        binding: binding_spec.slot.binding,
                        views,
                    });
                }
                (CustomBindingKind::Sampler, CustomBindingValue::Sampler(id)) => {
                    let Some(Some(sampler)) = samplers.get(id.0 as usize) else {
                        return Err(anyhow!("custom draw sampler {} is missing", id.0));
                    };
                    owned_resources.push(OwnedBindingResource::Sampler {
                        binding: binding_spec.slot.binding,
                        sampler: sampler.clone(),
                    });
                }
                (_, value) => {
                    return Err(anyhow!(
                        "custom draw binding value {:?} does not match binding kind",
                        value
                    ));
                }
            }
        }

        let mut bind_groups = Vec::new();

        for (group, mut owned_resources) in owned_resources_by_group {
            owned_resources.sort_by_key(|resource| match resource {
                OwnedBindingResource::Buffer { binding, .. }
                | OwnedBindingResource::BufferArray { binding, .. }
                | OwnedBindingResource::Texture { binding, .. }
                | OwnedBindingResource::TextureArray { binding, .. }
                | OwnedBindingResource::Sampler { binding, .. } => *binding,
            });

            let mut buffer_array_entries_by_binding = BTreeMap::new();
            let mut texture_array_entries_by_binding = BTreeMap::new();
            for resource in &owned_resources {
                match resource {
                    OwnedBindingResource::BufferArray { binding, elements } => {
                        let array: Vec<wgpu::BufferBinding<'_>> = elements
                            .iter()
                            .map(|element| wgpu::BufferBinding {
                                buffer: &element.buffer,
                                offset: element.offset,
                                size: element.size,
                            })
                            .collect();
                        buffer_array_entries_by_binding.insert(*binding, array);
                    }
                    OwnedBindingResource::TextureArray { binding, views } => {
                        let array: Vec<&wgpu::TextureView> = views.iter().collect();
                        texture_array_entries_by_binding.insert(*binding, array);
                    }
                    _ => {}
                }
            }

            let mut bind_group_entries = Vec::with_capacity(owned_resources.len());
            for resource in &owned_resources {
                match resource {
                    OwnedBindingResource::Buffer {
                        binding,
                        buffer,
                        offset,
                        size,
                    } => bind_group_entries.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: *offset,
                            size: *size,
                        }),
                    }),
                    OwnedBindingResource::BufferArray { binding, .. } => {
                        let Some(array) = buffer_array_entries_by_binding.get(binding) else {
                            return Err(anyhow!(
                                "custom draw buffer binding array {} is missing",
                                binding
                            ));
                        };
                        bind_group_entries.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::BufferArray(array.as_slice()),
                        });
                    }
                    OwnedBindingResource::Texture { binding, view } => {
                        bind_group_entries.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::TextureView(view),
                        });
                    }
                    OwnedBindingResource::TextureArray { binding, .. } => {
                        let Some(array) = texture_array_entries_by_binding.get(binding) else {
                            return Err(anyhow!(
                                "custom draw texture binding array {} is missing",
                                binding
                            ));
                        };
                        bind_group_entries.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::TextureViewArray(array.as_slice()),
                        });
                    }
                    OwnedBindingResource::Sampler { binding, sampler } => {
                        bind_group_entries.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        });
                    }
                }
            }

            let Some(bind_group_layout) = bind_group_layouts.get(group as usize) else {
                return Err(anyhow!("custom draw bind group {} is out of range", group));
            };

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bind_group_layout,
                entries: &bind_group_entries,
            });

            bind_groups.push((group, bind_group));
        }

        Ok(bind_groups)
    }

    fn resolve_buffer_resource(
        &self,
        binding: u32,
        source: &CustomBufferSource,
        buffers: &[Option<WgpuCustomBuffer>],
        usage: wgpu::BufferUsages,
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<OwnedBindingResource> {
        match source {
            CustomBufferSource::Buffer(id) => {
                let Some(Some(buffer_entry)) = buffers.get(id.0 as usize) else {
                    return Err(anyhow!("custom draw buffer {} is missing", id.0));
                };
                Ok(OwnedBindingResource::Buffer {
                    binding,
                    buffer: buffer_entry.buffer.clone(),
                    offset: 0,
                    size: NonZeroU64::new(buffer_entry.size),
                })
            }
            CustomBufferSource::BufferSlice { id, offset, size } => {
                let Some(Some(buffer_entry)) = buffers.get(id.0 as usize) else {
                    return Err(anyhow!("custom draw buffer {} is missing", id.0));
                };
                let end = offset
                    .checked_add(*size)
                    .ok_or_else(|| anyhow!("custom draw buffer slice overflow"))?;
                if end > buffer_entry.size {
                    return Err(anyhow!(
                        "custom draw buffer slice out of bounds (offset {} size {} buffer {})",
                        offset,
                        size,
                        buffer_entry.size
                    ));
                }
                Ok(OwnedBindingResource::Buffer {
                    binding,
                    buffer: buffer_entry.buffer.clone(),
                    offset: *offset,
                    size: NonZeroU64::new(*size),
                })
            }
            CustomBufferSource::Inline(data) => {
                let inline_buffer = self.create_inline_buffer(data, usage, "custom_inline_binding");
                let inline_size = (data.len() as u64).max(4);
                temporary_buffers.push(inline_buffer.clone());
                Ok(OwnedBindingResource::Buffer {
                    binding,
                    buffer: inline_buffer,
                    offset: 0,
                    size: NonZeroU64::new(inline_size),
                })
            }
        }
    }

    fn create_inline_buffer(
        &self,
        data: &[u8],
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> wgpu::Buffer {
        let buffer_size = (data.len() as u64).max(4);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size,
            usage,
            mapped_at_creation: true,
        });

        {
            let mut mapped_range = buffer.slice(..).get_mapped_range_mut();
            let copy_len = data.len().min(mapped_range.len());
            mapped_range
                .slice(..copy_len)
                .copy_from_slice(&data[..copy_len]);
        }

        buffer.unmap();
        buffer
    }

    fn create_registered_buffer(&self, data: &[u8], label: &str) -> WgpuCustomBuffer {
        let buffer_size = (data.len() as u64).max(4);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        {
            let mut mapped_range = buffer.slice(..).get_mapped_range_mut();
            let copy_len = data.len().min(mapped_range.len());
            mapped_range
                .slice(..copy_len)
                .copy_from_slice(&data[..copy_len]);
        }

        buffer.unmap();

        WgpuCustomBuffer {
            buffer,
            size: buffer_size,
        }
    }

    fn create_render_pipeline(&self, desc: CustomPipelineDesc) -> Result<WgpuCustomPipeline> {
        if desc.state.sample_count == 0
            || desc.state.sample_count > MAX_SAMPLE_COUNT
            || !desc.state.sample_count.is_power_of_two()
        {
            return Err(anyhow!(
                "custom draw sample count must be a power of two between 1 and {} (got {})",
                MAX_SAMPLE_COUNT,
                desc.state.sample_count
            ));
        }

        let device_features = self.device.features();
        for binding in &desc.bindings {
            match binding.kind {
                CustomBindingKind::Buffer
                | CustomBindingKind::BufferArray { .. }
                | CustomBindingKind::Texture
                | CustomBindingKind::TextureArray { .. }
                | CustomBindingKind::StorageTexture
                | CustomBindingKind::StorageTextureArray { .. }
                | CustomBindingKind::Sampler
                | CustomBindingKind::Uniform { .. } => {}
            }

            let required_features = binding_kind_required_features(binding.kind);
            if !device_features.contains(required_features) {
                return Err(anyhow!(
                    "custom draw binding {:?} requires unsupported wgpu features: {:?}",
                    binding.kind,
                    required_features
                ));
            }
        }

        let mut module = naga::front::wgsl::parse_str(&desc.shader_source)
            .map_err(|error| anyhow!("custom draw WGSL parse failed: {error}"))?;
        let validator_flags =
            naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let mut info =
            naga::valid::Validator::new(validator_flags, naga::valid::Capabilities::empty())
                .validate(&module)
                .map_err(|error| anyhow!("custom draw WGSL validation failed: {error}"))?;

        let vertex_entry_index = module
            .entry_points
            .iter()
            .position(|entry| {
                entry.stage == naga::ShaderStage::Vertex && entry.name == desc.vertex_entry
            })
            .ok_or_else(|| anyhow!("custom draw vertex entry '{}' not found", desc.vertex_entry))?;

        let fragment_entry_index = module
            .entry_points
            .iter()
            .position(|entry| {
                entry.stage == naga::ShaderStage::Fragment && entry.name == desc.fragment_entry
            })
            .ok_or_else(|| {
                anyhow!(
                    "custom draw fragment entry '{}' not found",
                    desc.fragment_entry
                )
            })?;

        let push_constants_slot = push_constants_slot(&desc.bindings);
        let push_constants = apply_push_constants(
            &mut module,
            &info,
            &[vertex_entry_index, fragment_entry_index],
            desc.push_constants,
            push_constants_slot,
        )?;
        if push_constants.is_some() {
            info = naga::valid::Validator::new(validator_flags, naga::valid::Capabilities::empty())
                .validate(&module)
                .map_err(|error| anyhow!("custom draw WGSL validation failed: {error}"))?;
        }

        let attribute_locations = build_attribute_locations(&desc.vertex_fetches)?;
        assign_vertex_locations(&mut module, vertex_entry_index, &attribute_locations)?;

        let (mut bindings_by_name, bindings_by_slot) = build_binding_maps(&desc.bindings);
        if let Some(push_constants) = &push_constants {
            if bindings_by_name.contains_key(push_constants.name) {
                return Err(anyhow!(
                    "custom draw push constants name '{}' conflicts with a binding name",
                    push_constants.name
                ));
            }
            bindings_by_name.insert(
                push_constants.name,
                BindingInfo {
                    kind: CustomBindingKind::Uniform {
                        size: push_constants.size,
                    },
                    slot: push_constants.slot,
                },
            );
        }
        let vertex_entry_name = module.entry_points[vertex_entry_index].name.clone();
        let fragment_entry_name = module.entry_points[fragment_entry_index].name.clone();

        assign_resource_bindings(
            &mut module,
            &info,
            &vertex_entry_name,
            vertex_entry_index,
            &bindings_by_name,
            &bindings_by_slot,
        )?;

        assign_resource_bindings(
            &mut module,
            &info,
            &fragment_entry_name,
            fragment_entry_index,
            &bindings_by_name,
            &bindings_by_slot,
        )?;

        info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|error| anyhow!("custom draw WGSL validation failed: {error}"))?;

        let mut sampled_texture_infos =
            collect_sampled_texture_binding_info(&module, &info, vertex_entry_index)?;
        merge_sampled_texture_binding_infos(
            &mut sampled_texture_infos,
            collect_sampled_texture_binding_info(&module, &info, fragment_entry_index)?,
        )?;

        let mut storage_texture_infos =
            collect_storage_texture_binding_info(&module, &info, vertex_entry_index)?;
        merge_storage_texture_binding_infos(
            &mut storage_texture_infos,
            collect_storage_texture_binding_info(&module, &info, fragment_entry_index)?,
        )?;

        let rewritten_wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .map_err(|error| anyhow!("custom draw WGSL rewrite failed: {error}"))?;

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("custom_draw_shader"),
                source: wgpu::ShaderSource::Wgsl(rewritten_wgsl.into()),
            });

        let mut binding_specs =
            Vec::with_capacity(desc.bindings.len() + usize::from(push_constants.is_some()));
        let mut bind_group_layout_entries_by_group: Vec<Vec<wgpu::BindGroupLayoutEntry>> =
            Vec::new();

        for binding in &desc.bindings {
            let slot = binding.slot.unwrap_or(CustomBindingSlot {
                group: 0,
                binding: binding.name.index(),
            });
            binding_specs.push(WgpuBindingSpec {
                kind: binding.kind,
                slot,
            });

            if bind_group_layout_entries_by_group.len() <= slot.group as usize {
                bind_group_layout_entries_by_group.resize_with(slot.group as usize + 1, Vec::new);
            }

            bind_group_layout_entries_by_group[slot.group as usize].push(
                wgpu::BindGroupLayoutEntry {
                    binding: slot.binding,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: map_binding_type(
                        binding.kind,
                        storage_texture_infos
                            .get(&(slot.group, slot.binding))
                            .copied(),
                        sampled_texture_infos
                            .get(&(slot.group, slot.binding))
                            .copied(),
                    )?,
                    count: binding_array_count(binding.kind),
                },
            );
        }

        if let Some(push_constants) = &push_constants {
            binding_specs.push(WgpuBindingSpec {
                kind: CustomBindingKind::Uniform {
                    size: push_constants.size,
                },
                slot: push_constants.slot,
            });

            if bind_group_layout_entries_by_group.len() <= push_constants.slot.group as usize {
                bind_group_layout_entries_by_group
                    .resize_with(push_constants.slot.group as usize + 1, Vec::new);
            }

            bind_group_layout_entries_by_group[push_constants.slot.group as usize].push(
                wgpu::BindGroupLayoutEntry {
                    binding: push_constants.slot.binding,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: map_binding_type(
                        CustomBindingKind::Uniform {
                            size: push_constants.size,
                        },
                        None,
                        None,
                    )?,
                    count: None,
                },
            );
        }

        let mut bind_group_layouts = Vec::with_capacity(bind_group_layout_entries_by_group.len());
        for (group_index, mut entries) in bind_group_layout_entries_by_group.into_iter().enumerate()
        {
            entries.sort_by_key(|entry| entry.binding);
            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("custom_draw_bind_group_layout_{}", group_index)),
                        entries: &entries,
                    });
            bind_group_layouts.push(bind_group_layout);
        }

        let bind_group_layout_refs: Vec<Option<&wgpu::BindGroupLayout>> =
            bind_group_layouts.iter().map(Some).collect();

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("custom_draw_pipeline_layout"),
                bind_group_layouts: &bind_group_layout_refs,
                immediate_size: 0,
            });

        let mut vertex_attribute_layouts = Vec::with_capacity(desc.vertex_fetches.len());
        for fetch in &desc.vertex_fetches {
            let mut attributes = Vec::with_capacity(fetch.layout.attributes.len());
            for attribute in &fetch.layout.attributes {
                let location = attribute
                    .location
                    .or_else(|| attribute_locations.get(attribute.name.as_str()).copied())
                    .ok_or_else(|| {
                        anyhow!(
                            "custom draw vertex attribute '{}' has no assigned location",
                            attribute.name.as_str()
                        )
                    })?;
                attributes.push(wgpu::VertexAttribute {
                    format: map_vertex_format(attribute.format)?,
                    offset: u64::from(attribute.offset),
                    shader_location: location,
                });
            }
            vertex_attribute_layouts.push(attributes);
        }

        let mut vertex_buffer_layouts = Vec::with_capacity(desc.vertex_fetches.len());
        for (fetch, attributes) in desc
            .vertex_fetches
            .iter()
            .zip(vertex_attribute_layouts.iter())
        {
            vertex_buffer_layouts.push(wgpu::VertexBufferLayout {
                array_stride: u64::from(fetch.layout.stride),
                step_mode: if fetch.instanced {
                    wgpu::VertexStepMode::Instance
                } else {
                    wgpu::VertexStepMode::Vertex
                },
                attributes,
            });
        }

        let color_formats = if desc.color_targets.is_empty() {
            vec![self.surface_format]
        } else {
            let mut formats = Vec::with_capacity(desc.color_targets.len());
            for color_target in &desc.color_targets {
                let Some(format) = map_texture_format(*color_target) else {
                    return Err(anyhow!(
                        "custom draw color target format {:?} is not supported on this wgpu renderer",
                        color_target
                    ));
                };
                formats.push(format);
            }
            formats
        };

        let mut color_targets = Vec::with_capacity(color_formats.len());
        for format in &color_formats {
            color_targets.push(Some(wgpu::ColorTargetState {
                format: *format,
                blend: map_blend_state(desc.state.blend),
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }

        let depth_stencil = desc.state.depth.map(|depth_state| wgpu::DepthStencilState {
            format: map_depth_format(depth_state.format),
            depth_write_enabled: Some(depth_state.write_enabled),
            depth_compare: Some(map_depth_compare(depth_state.compare)),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let depth_format = depth_stencil.as_ref().map(|state| state.format);

        let render_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&desc.name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some(&desc.vertex_entry),
                    buffers: &vertex_buffer_layouts,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some(&desc.fragment_entry),
                    targets: &color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: map_primitive_topology(desc.primitive),
                    strip_index_format: None,
                    front_face: map_front_face(desc.state.front_face),
                    cull_mode: map_cull_mode(desc.state.cull_mode),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: desc.state.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        Ok(WgpuCustomPipeline {
            pipeline: render_pipeline,
            bind_group_layouts,
            bindings: binding_specs,
            color_formats,
            sample_count: desc.state.sample_count,
            depth_format,
            vertex_fetch_count: desc.vertex_fetches.len(),
        })
    }

    fn create_compute_pipeline_impl(
        &self,
        desc: CustomComputePipelineDesc,
    ) -> Result<WgpuCustomComputePipeline> {
        let device_features = self.device.features();
        for binding in &desc.bindings {
            match binding.kind {
                CustomBindingKind::Buffer
                | CustomBindingKind::BufferArray { .. }
                | CustomBindingKind::Texture
                | CustomBindingKind::TextureArray { .. }
                | CustomBindingKind::Sampler
                | CustomBindingKind::Uniform { .. }
                | CustomBindingKind::StorageTexture
                | CustomBindingKind::StorageTextureArray { .. } => {}
            }

            let required_features = binding_kind_required_features(binding.kind);
            if !device_features.contains(required_features) {
                return Err(anyhow!(
                    "custom compute binding {:?} requires unsupported wgpu features: {:?}",
                    binding.kind,
                    required_features
                ));
            }
        }

        let mut module = naga::front::wgsl::parse_str(&desc.shader_source)
            .map_err(|error| anyhow!("custom compute WGSL parse failed: {error}"))?;
        let validator_flags =
            naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let mut info =
            naga::valid::Validator::new(validator_flags, naga::valid::Capabilities::empty())
                .validate(&module)
                .map_err(|error| anyhow!("custom compute WGSL validation failed: {error}"))?;

        let compute_entry_index = module
            .entry_points
            .iter()
            .position(|entry| {
                entry.stage == naga::ShaderStage::Compute && entry.name == desc.entry_point
            })
            .ok_or_else(|| anyhow!("custom compute entry '{}' not found", desc.entry_point))?;

        let push_constants_slot = push_constants_slot(&desc.bindings);
        let push_constants = apply_push_constants(
            &mut module,
            &info,
            &[compute_entry_index],
            desc.push_constants,
            push_constants_slot,
        )?;
        if push_constants.is_some() {
            info = naga::valid::Validator::new(validator_flags, naga::valid::Capabilities::empty())
                .validate(&module)
                .map_err(|error| anyhow!("custom compute WGSL validation failed: {error}"))?;
        }

        let (mut bindings_by_name, bindings_by_slot) = build_binding_maps(&desc.bindings);
        if let Some(push_constants) = &push_constants {
            if bindings_by_name.contains_key(push_constants.name) {
                return Err(anyhow!(
                    "custom compute push constants name '{}' conflicts with a binding name",
                    push_constants.name
                ));
            }
            bindings_by_name.insert(
                push_constants.name,
                BindingInfo {
                    kind: CustomBindingKind::Uniform {
                        size: push_constants.size,
                    },
                    slot: push_constants.slot,
                },
            );
        }
        let compute_entry_name = module.entry_points[compute_entry_index].name.clone();

        assign_resource_bindings(
            &mut module,
            &info,
            &compute_entry_name,
            compute_entry_index,
            &bindings_by_name,
            &bindings_by_slot,
        )?;

        info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|error| anyhow!("custom compute WGSL validation failed: {error}"))?;

        let storage_texture_infos =
            collect_storage_texture_binding_info(&module, &info, compute_entry_index)?;
        let sampled_texture_infos =
            collect_sampled_texture_binding_info(&module, &info, compute_entry_index)?;

        let rewritten_wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .map_err(|error| anyhow!("custom compute WGSL rewrite failed: {error}"))?;

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("custom_compute_shader"),
                source: wgpu::ShaderSource::Wgsl(rewritten_wgsl.into()),
            });

        let mut binding_specs =
            Vec::with_capacity(desc.bindings.len() + usize::from(push_constants.is_some()));
        let mut bind_group_layout_entries_by_group: Vec<Vec<wgpu::BindGroupLayoutEntry>> =
            Vec::new();

        for binding in &desc.bindings {
            let slot = binding.slot.unwrap_or(CustomBindingSlot {
                group: 0,
                binding: binding.name.index(),
            });
            binding_specs.push(WgpuBindingSpec {
                kind: binding.kind,
                slot,
            });

            let binding_type = map_binding_type(
                binding.kind,
                storage_texture_infos
                    .get(&(slot.group, slot.binding))
                    .copied(),
                sampled_texture_infos
                    .get(&(slot.group, slot.binding))
                    .copied(),
            )?;

            if bind_group_layout_entries_by_group.len() <= slot.group as usize {
                bind_group_layout_entries_by_group.resize_with(slot.group as usize + 1, Vec::new);
            }

            bind_group_layout_entries_by_group[slot.group as usize].push(
                wgpu::BindGroupLayoutEntry {
                    binding: slot.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: binding_type,
                    count: binding_array_count(binding.kind),
                },
            );
        }

        if let Some(push_constants) = &push_constants {
            binding_specs.push(WgpuBindingSpec {
                kind: CustomBindingKind::Uniform {
                    size: push_constants.size,
                },
                slot: push_constants.slot,
            });

            if bind_group_layout_entries_by_group.len() <= push_constants.slot.group as usize {
                bind_group_layout_entries_by_group
                    .resize_with(push_constants.slot.group as usize + 1, Vec::new);
            }

            bind_group_layout_entries_by_group[push_constants.slot.group as usize].push(
                wgpu::BindGroupLayoutEntry {
                    binding: push_constants.slot.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: map_binding_type(
                        CustomBindingKind::Uniform {
                            size: push_constants.size,
                        },
                        None,
                        None,
                    )?,
                    count: None,
                },
            );
        }

        let mut bind_group_layouts = Vec::with_capacity(bind_group_layout_entries_by_group.len());
        for (group_index, mut entries) in bind_group_layout_entries_by_group.into_iter().enumerate()
        {
            entries.sort_by_key(|entry| entry.binding);
            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("custom_compute_bind_group_layout_{}", group_index)),
                        entries: &entries,
                    });
            bind_group_layouts.push(bind_group_layout);
        }

        let bind_group_layout_refs: Vec<Option<&wgpu::BindGroupLayout>> =
            bind_group_layouts.iter().map(Some).collect();

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("custom_compute_pipeline_layout"),
                bind_group_layouts: &bind_group_layout_refs,
                immediate_size: 0,
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&desc.name),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some(&desc.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        Ok(WgpuCustomComputePipeline {
            pipeline: compute_pipeline,
            bind_group_layouts,
            bindings: binding_specs,
        })
    }

    fn texture_entry_bytes(texture: &WgpuCustomTexture) -> u64 {
        let block = texture.format.block_info();
        let mut total_bytes = 0u64;

        for mip_level in 0..texture.mip_level_count {
            let mip_width = (texture.width >> mip_level).max(1);
            let mip_height = (texture.height >> mip_level).max(1);
            let blocks_x = mip_width.div_ceil(block.width);
            let blocks_y = mip_height.div_ceil(block.height);
            total_bytes = total_bytes.saturating_add(
                u64::from(blocks_x)
                    .saturating_mul(u64::from(blocks_y))
                    .saturating_mul(u64::from(block.bytes))
                    .saturating_mul(u64::from(texture.array_layer_count)),
            );
        }

        if texture.is_render_target && texture.sample_count > 1 && texture.msaa_texture.is_some() {
            let blocks_x = texture.width.div_ceil(block.width);
            let blocks_y = texture.height.div_ceil(block.height);
            let msaa_bytes = u64::from(blocks_x)
                .saturating_mul(u64::from(blocks_y))
                .saturating_mul(u64::from(block.bytes))
                .saturating_mul(u64::from(texture.array_layer_count))
                .saturating_mul(u64::from(texture.sample_count));
            total_bytes = total_bytes.saturating_add(msaa_bytes);
        }

        total_bytes
    }

    fn upload_texture_level(
        &self,
        texture: &WgpuCustomTexture,
        level: u32,
        data: &[u8],
        bytes_per_row: Option<u32>,
    ) -> Result<()> {
        if level >= texture.mip_level_count {
            return Err(anyhow!(
                "custom texture mip level {} out of bounds (max {})",
                level,
                texture.mip_level_count.saturating_sub(1)
            ));
        }

        let block = texture.format.block_info();
        let mip_width = (texture.width >> level).max(1);
        let mip_height = (texture.height >> level).max(1);
        if texture.format.is_compressed()
            && (!mip_width.is_multiple_of(block.width) || !mip_height.is_multiple_of(block.height))
        {
            return Err(anyhow!(
                "compressed custom texture mip dimensions must be multiples of block size {}x{} (got {}x{})",
                block.width,
                block.height,
                mip_width,
                mip_height
            ));
        }
        let rows_per_image = mip_height.div_ceil(block.height);
        let packed_bytes_per_row = mip_width.div_ceil(block.width).saturating_mul(block.bytes);

        let upload_bytes_per_row = bytes_per_row.unwrap_or(packed_bytes_per_row);
        if upload_bytes_per_row < packed_bytes_per_row {
            return Err(anyhow!(
                "custom texture bytes_per_row {} is smaller than packed row size {}",
                upload_bytes_per_row,
                packed_bytes_per_row
            ));
        }
        if !upload_bytes_per_row.is_multiple_of(block.bytes) {
            return Err(anyhow!(
                "custom texture bytes_per_row {} is not a multiple of block size {}",
                upload_bytes_per_row,
                block.bytes
            ));
        }

        let required_bytes = u64::from(upload_bytes_per_row)
            .saturating_mul(u64::from(rows_per_image))
            .saturating_mul(u64::from(texture.array_layer_count));
        if required_bytes > data.len() as u64 {
            return Err(anyhow!(
                "custom texture upload is too small (need {} bytes, got {})",
                required_bytes,
                data.len()
            ));
        }

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload_bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width: mip_width,
                height: mip_height,
                depth_or_array_layers: texture.array_layer_count,
            },
        );

        Ok(())
    }
}

impl CustomDrawRegistry for WgpuCustomDrawRegistry {
    fn create_pipeline(&self, desc: CustomPipelineDesc) -> Result<CustomPipelineId> {
        let pipeline = self.create_render_pipeline(desc)?;
        let mut pipelines = self.pipelines.lock();
        let pipeline_id = alloc_slot(&mut pipelines, pipeline);
        Ok(CustomPipelineId(pipeline_id))
    }

    fn create_pipeline_msl(
        &self,
        _desc: CustomPipelineDesc,
        _msl_source: String,
    ) -> Result<CustomPipelineId> {
        Err(anyhow!(
            "custom draw MSL source pipelines are only supported on Metal"
        ))
    }

    fn create_pipeline_metallib(
        &self,
        _desc: CustomPipelineDesc,
        _metallib_data: Arc<[u8]>,
    ) -> Result<CustomPipelineId> {
        Err(anyhow!(
            "custom draw metallib pipelines are only supported on Metal"
        ))
    }

    fn set_pipeline_cache_path(&self, _path: Option<PathBuf>) -> Result<()> {
        Err(anyhow!(
            "custom draw pipeline cache path is only supported on Metal"
        ))
    }

    fn set_gpu_profiling_enabled(&self, enabled: bool) -> Result<()> {
        let mut profiling = self.profiling.lock();
        profiling.gpu_profiling_enabled = enabled;
        if !enabled {
            profiling.last_gpu_profile = None;
            if !profiling.frame_diagnostics_enabled {
                profiling.last_gpu_time_ns = None;
            }
        }
        Ok(())
    }

    fn take_last_gpu_profile(&self) -> Option<CustomGpuFrameProfile> {
        self.profiling.lock().last_gpu_profile.take()
    }

    fn set_frame_diagnostics_enabled(&self, enabled: bool) -> Result<()> {
        let mut profiling = self.profiling.lock();
        profiling.frame_diagnostics_enabled = enabled;
        if !enabled {
            profiling.last_frame_diagnostics = None;
            if !profiling.gpu_profiling_enabled {
                profiling.last_gpu_time_ns = None;
            }
        }
        Ok(())
    }

    fn take_last_frame_diagnostics(&self) -> Option<CustomFrameDiagnostics> {
        self.profiling.lock().last_frame_diagnostics.take()
    }

    fn resource_stats(&self) -> CustomDrawResourceStats {
        let pipelines = self.pipelines.lock();
        let compute_pipelines = self.compute_pipelines.lock();
        let buffers = self.buffers.lock();
        let textures = self.textures.lock();
        let depth_targets = self.depth_targets.lock();
        let samplers = self.samplers.lock();

        let buffer_bytes = buffers
            .iter()
            .filter_map(|entry| entry.as_ref().map(|buffer| buffer.size))
            .sum();

        let texture_bytes = textures
            .iter()
            .filter_map(|entry| entry.as_ref().map(Self::texture_entry_bytes))
            .sum();

        let render_target_count = textures
            .iter()
            .filter_map(|entry| entry.as_ref())
            .filter(|entry| entry.is_render_target)
            .count() as u32;

        let depth_target_bytes = depth_targets
            .iter()
            .filter_map(|entry| entry.as_ref())
            .map(|entry| depth_target_estimate_bytes(entry.width, entry.height, entry.sample_count))
            .sum();
        let depth_target_count =
            depth_targets.iter().filter(|entry| entry.is_some()).count() as u32;

        CustomDrawResourceStats {
            pipeline_count: pipelines.iter().filter(|entry| entry.is_some()).count() as u32,
            compute_pipeline_count: compute_pipelines
                .iter()
                .filter(|entry| entry.is_some())
                .count() as u32,
            buffer_count: buffers.iter().filter(|entry| entry.is_some()).count() as u32,
            buffer_bytes,
            texture_count: textures.iter().filter(|entry| entry.is_some()).count() as u32,
            texture_bytes,
            render_target_count,
            depth_target_count,
            depth_target_bytes,
            sampler_count: samplers.iter().filter(|entry| entry.is_some()).count() as u32,
        }
    }

    fn texture_format_supported(&self, format: CustomTextureFormat) -> bool {
        let Some(texture_format) = map_texture_format(format) else {
            return false;
        };
        texture_format_supported_by_device(self.device.features(), texture_format)
    }

    fn create_compute_pipeline(
        &self,
        desc: CustomComputePipelineDesc,
    ) -> Result<CustomComputePipelineId> {
        let pipeline = self.create_compute_pipeline_impl(desc)?;
        let mut compute_pipelines = self.compute_pipelines.lock();
        let pipeline_id = alloc_slot(&mut compute_pipelines, pipeline);
        Ok(CustomComputePipelineId(pipeline_id))
    }

    fn create_buffer(&self, desc: CustomBufferDesc) -> Result<CustomBufferId> {
        let buffer = self.create_registered_buffer(&desc.data, &desc.name);
        let mut buffers = self.buffers.lock();
        let buffer_id = alloc_slot(&mut buffers, buffer);
        Ok(CustomBufferId(buffer_id))
    }

    fn update_buffer(&self, id: CustomBufferId, data: Arc<[u8]>) -> Result<()> {
        let mut buffers = self.buffers.lock();
        let Some(slot) = buffers.get_mut(id.0 as usize) else {
            return Err(anyhow!("custom draw buffer {} not found", id.0));
        };
        let Some(buffer_entry) = slot.as_mut() else {
            return Err(anyhow!("custom draw buffer {} not found", id.0));
        };

        if (data.len() as u64) <= buffer_entry.size {
            self.queue.write_buffer(&buffer_entry.buffer, 0, &data);
            return Ok(());
        }

        let replacement = self.create_registered_buffer(&data, "custom_buffer_resize");
        *buffer_entry = replacement;
        Ok(())
    }

    fn remove_buffer(&self, id: CustomBufferId) {
        let mut buffers = self.buffers.lock();
        if let Some(slot) = buffers.get_mut(id.0 as usize) {
            slot.take();
        }
    }

    fn create_texture(&self, desc: CustomTextureDesc) -> Result<CustomTextureId> {
        let Some(texture_format) = map_texture_format(desc.format) else {
            return Err(anyhow!(
                "custom texture format {:?} is not supported by this wgpu renderer",
                desc.format
            ));
        };

        let device_features = self.device.features();
        if !texture_format_supported_by_device(device_features, texture_format) {
            return Err(anyhow!(
                "custom texture format {:?} is not supported by this wgpu renderer",
                desc.format
            ));
        }

        if matches!(desc.dimension, CustomTextureDimension::Cube) && desc.width != desc.height {
            return Err(anyhow!(
                "custom cube textures require width and height to match"
            ));
        }

        let array_layer_count = desc.dimension.array_layers();
        let view_dimension = map_texture_view_dimension(desc.dimension)?;

        let mut texture_usage = wgpu::TextureUsages::COPY_DST;
        if desc.usage.contains(CustomTextureUsage::SAMPLED) {
            texture_usage |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        if desc.usage.contains(CustomTextureUsage::STORAGE) {
            if desc.dimension.is_array() {
                return Err(anyhow!("custom storage textures must be 2D"));
            }
            let Some(storage_format) = map_custom_storage_texture_format(desc.format) else {
                return Err(anyhow!(
                    "custom texture format {:?} is not supported for storage usage on wgpu",
                    desc.format
                ));
            };
            if storage_format != texture_format {
                return Err(anyhow!(
                    "custom texture format {:?} cannot be used as a storage texture on wgpu",
                    desc.format
                ));
            }
            if texture_format == wgpu::TextureFormat::Bgra8Unorm
                && !device_features.contains(wgpu::Features::BGRA8UNORM_STORAGE)
            {
                return Err(anyhow!(
                    "custom texture format {:?} requires BGRA8 storage support on this wgpu renderer",
                    desc.format
                ));
            }
            texture_usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        if texture_usage == wgpu::TextureUsages::COPY_DST {
            return Err(anyhow!(
                "custom texture usage must include sampled and/or storage usage"
            ));
        }

        let width = desc.width.max(1);
        let height = desc.height.max(1);
        let mip_level_count = desc.data.len().max(1) as u32;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&desc.name),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layer_count,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: texture_usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dimension),
            ..Default::default()
        });

        let texture_entry = WgpuCustomTexture {
            texture,
            view,
            msaa_texture: None,
            msaa_view: None,
            width,
            height,
            array_layer_count,
            mip_level_count,
            sample_count: 1,
            format: desc.format,
            clear_color: [0.0; 4],
            is_render_target: false,
        };

        for (level, data) in desc.data.iter().enumerate() {
            self.upload_texture_level(&texture_entry, level as u32, data, None)?;
        }

        let mut textures = self.textures.lock();
        let texture_id = alloc_slot(&mut textures, texture_entry);
        Ok(CustomTextureId(texture_id))
    }

    fn create_render_target(&self, desc: CustomRenderTargetDesc) -> Result<CustomTextureId> {
        if desc.format.is_compressed() {
            return Err(anyhow!(
                "custom render targets must not use compressed formats"
            ));
        }
        if desc.sample_count == 0
            || desc.sample_count > MAX_SAMPLE_COUNT
            || !desc.sample_count.is_power_of_two()
        {
            return Err(anyhow!(
                "custom draw render target sample count must be a power of two between 1 and {} (got {})",
                MAX_SAMPLE_COUNT,
                desc.sample_count
            ));
        }

        let Some(texture_format) = map_texture_format(desc.format) else {
            return Err(anyhow!(
                "custom render target format {:?} is not supported by this wgpu renderer",
                desc.format
            ));
        };

        let width = desc.width.max(1);
        let height = desc.height.max(1);

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&desc.name),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (msaa_texture, msaa_view) = if desc.sample_count > 1 {
            let msaa_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{}_msaa", desc.name)),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: desc.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: texture_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(msaa_texture), Some(msaa_view))
        } else {
            (None, None)
        };

        let texture_entry = WgpuCustomTexture {
            texture,
            view,
            msaa_texture,
            msaa_view,
            width,
            height,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: desc.sample_count,
            format: desc.format,
            clear_color: desc.clear_color.unwrap_or([0.0, 0.0, 0.0, 0.0]),
            is_render_target: true,
        };

        let mut textures = self.textures.lock();
        let texture_id = alloc_slot(&mut textures, texture_entry);
        Ok(CustomTextureId(texture_id))
    }

    fn update_texture(&self, id: CustomTextureId, update: CustomTextureUpdate) -> Result<()> {
        let textures = self.textures.lock();
        let Some(Some(texture_entry)) = textures.get(id.0 as usize) else {
            return Err(anyhow!("custom draw texture {} not found", id.0));
        };
        if texture_entry.is_render_target {
            return Err(anyhow!("custom render targets cannot be updated"));
        }
        self.upload_texture_level(
            texture_entry,
            update.level,
            &update.data,
            update.bytes_per_row,
        )
    }

    fn update_texture_from_buffer(
        &self,
        id: CustomTextureId,
        update: CustomTextureBufferUpdate,
    ) -> Result<()> {
        let CustomTextureBufferUpdate {
            level,
            buffer,
            bytes_per_row,
        } = update;

        let texture_entry = {
            let textures = self.textures.lock();
            let Some(Some(texture_entry)) = textures.get(id.0 as usize) else {
                return Err(anyhow!("custom draw texture {} not found", id.0));
            };
            if texture_entry.is_render_target {
                return Err(anyhow!("custom render targets cannot be updated"));
            }
            texture_entry.clone()
        };

        if level >= texture_entry.mip_level_count {
            return Err(anyhow!(
                "custom texture mip level {} out of bounds (max {})",
                level,
                texture_entry.mip_level_count.saturating_sub(1)
            ));
        }

        let block = texture_entry.format.block_info();
        let mip_width = (texture_entry.width >> level).max(1);
        let mip_height = (texture_entry.height >> level).max(1);
        if texture_entry.format.is_compressed()
            && (!mip_width.is_multiple_of(block.width) || !mip_height.is_multiple_of(block.height))
        {
            return Err(anyhow!(
                "compressed custom texture mip dimensions must be multiples of block size {}x{} (got {}x{})",
                block.width,
                block.height,
                mip_width,
                mip_height
            ));
        }
        let rows_per_image = mip_height.div_ceil(block.height);
        let packed_bytes_per_row = mip_width.div_ceil(block.width).saturating_mul(block.bytes);
        let upload_bytes_per_row = bytes_per_row.unwrap_or(packed_bytes_per_row);

        if upload_bytes_per_row < packed_bytes_per_row {
            return Err(anyhow!(
                "custom texture bytes_per_row {} is smaller than packed row size {}",
                upload_bytes_per_row,
                packed_bytes_per_row
            ));
        }
        if !upload_bytes_per_row.is_multiple_of(block.bytes) {
            return Err(anyhow!(
                "custom texture bytes_per_row {} is not a multiple of block size {}",
                upload_bytes_per_row,
                block.bytes
            ));
        }

        let required_bytes = u64::from(upload_bytes_per_row)
            .saturating_mul(u64::from(rows_per_image))
            .saturating_mul(u64::from(texture_entry.array_layer_count));

        let (source_buffer, source_offset, source_size) = {
            let buffers = self.buffers.lock();
            match buffer {
                CustomBufferSource::Buffer(buffer_id) => {
                    let Some(Some(buffer_entry)) = buffers.get(buffer_id.0 as usize) else {
                        return Err(anyhow!("custom draw buffer {} is missing", buffer_id.0));
                    };
                    (buffer_entry.buffer.clone(), 0, buffer_entry.size)
                }
                CustomBufferSource::BufferSlice {
                    id: buffer_id,
                    offset,
                    size,
                } => {
                    let Some(Some(buffer_entry)) = buffers.get(buffer_id.0 as usize) else {
                        return Err(anyhow!("custom draw buffer {} is missing", buffer_id.0));
                    };
                    if size == 0 {
                        return Err(anyhow!("custom texture buffer slice is empty"));
                    }
                    let end = offset
                        .checked_add(size)
                        .ok_or_else(|| anyhow!("custom texture buffer slice overflow"))?;
                    if end > buffer_entry.size {
                        return Err(anyhow!("custom texture buffer slice out of bounds"));
                    }
                    (buffer_entry.buffer.clone(), offset, size)
                }
                CustomBufferSource::Inline(_) => {
                    return Err(anyhow!(
                        "custom texture updates from buffer require a buffer source"
                    ));
                }
            }
        };

        if required_bytes > source_size {
            return Err(anyhow!(
                "custom texture buffer upload is too small (need {} bytes, got {})",
                required_bytes,
                source_size
            ));
        }

        if !source_offset.is_multiple_of(u64::from(block.bytes)) {
            return Err(anyhow!(
                "custom texture buffer offset {} is not aligned to block size {}",
                source_offset,
                block.bytes
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("custom_texture_buffer_upload"),
            });

        if texture_entry.array_layer_count == 1
            && rows_per_image == 1
            && upload_bytes_per_row == packed_bytes_per_row
        {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &source_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: source_offset,
                        bytes_per_row: None,
                        rows_per_image: None,
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture_entry.texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: mip_width,
                    height: mip_height,
                    depth_or_array_layers: 1,
                },
            );
        } else if upload_bytes_per_row.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &source_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: source_offset,
                        bytes_per_row: Some(upload_bytes_per_row),
                        rows_per_image: Some(rows_per_image),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture_entry.texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: mip_width,
                    height: mip_height,
                    depth_or_array_layers: texture_entry.array_layer_count,
                },
            );
        } else {
            let row_size = u64::from(upload_bytes_per_row);
            for layer in 0..texture_entry.array_layer_count {
                for row in 0..rows_per_image {
                    let row_index = u64::from(layer)
                        .saturating_mul(u64::from(rows_per_image))
                        .saturating_add(u64::from(row));
                    let row_offset = source_offset
                        .checked_add(row_index.saturating_mul(row_size))
                        .ok_or_else(|| anyhow!("custom texture buffer row offset overflow"))?;
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: &source_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: row_offset,
                                bytes_per_row: None,
                                rows_per_image: None,
                            },
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture_entry.texture,
                            mip_level: level,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: row.saturating_mul(block.height),
                                z: layer,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: mip_width,
                            height: block.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    fn remove_texture(&self, id: CustomTextureId) {
        let mut textures = self.textures.lock();
        if let Some(slot) = textures.get_mut(id.0 as usize) {
            slot.take();
        }
    }

    fn create_depth_target(&self, desc: CustomDepthTargetDesc) -> Result<CustomDepthTargetId> {
        if desc.sample_count == 0
            || desc.sample_count > MAX_SAMPLE_COUNT
            || !desc.sample_count.is_power_of_two()
        {
            return Err(anyhow!(
                "custom draw depth target sample count must be a power of two between 1 and {} (got {})",
                MAX_SAMPLE_COUNT,
                desc.sample_count
            ));
        }

        let width = desc.width.max(1);
        let height = desc.height.max(1);
        let format = map_depth_format(desc.format);

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&desc.name),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: desc.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let entry = WgpuCustomDepthTarget {
            view,
            width,
            height,
            sample_count: desc.sample_count,
            clear_depth: desc.clear_depth.unwrap_or(1.0),
            format,
        };

        let mut depth_targets = self.depth_targets.lock();
        let target_id = alloc_slot(&mut depth_targets, entry);
        Ok(CustomDepthTargetId(target_id))
    }

    fn remove_depth_target(&self, id: CustomDepthTargetId) {
        let mut depth_targets = self.depth_targets.lock();
        if let Some(slot) = depth_targets.get_mut(id.0 as usize) {
            slot.take();
        }
    }

    fn create_sampler(&self, desc: CustomSamplerDesc) -> Result<CustomSamplerId> {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&desc.name),
            address_mode_u: map_address_mode(desc.address_modes[0]),
            address_mode_v: map_address_mode(desc.address_modes[1]),
            address_mode_w: map_address_mode(desc.address_modes[2]),
            mag_filter: map_filter(desc.mag_filter),
            min_filter: map_filter(desc.min_filter),
            mipmap_filter: map_mipmap_filter(desc.mipmap_filter),
            ..Default::default()
        });

        let mut samplers = self.samplers.lock();
        let sampler_id = alloc_slot(&mut samplers, sampler);
        Ok(CustomSamplerId(sampler_id))
    }

    fn remove_sampler(&self, id: CustomSamplerId) {
        let mut samplers = self.samplers.lock();
        if let Some(slot) = samplers.get_mut(id.0 as usize) {
            slot.take();
        }
    }
}

fn alloc_slot<T>(slots: &mut Vec<Option<T>>, value: T) -> u32 {
    if let Some((slot_index, slot)) = slots
        .iter_mut()
        .enumerate()
        .find(|(_, slot)| slot.is_none())
    {
        *slot = Some(value);
        return slot_index as u32;
    }

    let slot_index = slots.len() as u32;
    slots.push(Some(value));
    slot_index
}

fn map_binding_type(
    kind: CustomBindingKind,
    storage_texture_info: Option<WgpuStorageTextureBindingInfo>,
    sampled_texture_info: Option<WgpuSampledTextureBindingInfo>,
) -> Result<wgpu::BindingType> {
    match kind {
        CustomBindingKind::Buffer => Ok(wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        }),
        CustomBindingKind::Uniform { size } => Ok(wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size as u64),
        }),
        CustomBindingKind::Texture => {
            let sampled_texture_info =
                sampled_texture_info.unwrap_or(WgpuSampledTextureBindingInfo {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                });
            Ok(wgpu::BindingType::Texture {
                sample_type: sampled_texture_info.sample_type,
                view_dimension: sampled_texture_info.view_dimension,
                multisampled: sampled_texture_info.multisampled,
            })
        }
        CustomBindingKind::Sampler => Ok(wgpu::BindingType::Sampler(
            wgpu::SamplerBindingType::Filtering,
        )),
        CustomBindingKind::StorageTexture => {
            let Some(storage_texture_info) = storage_texture_info else {
                return Err(anyhow!(
                    "custom storage texture binding metadata is missing"
                ));
            };
            Ok(wgpu::BindingType::StorageTexture {
                access: storage_texture_info.access,
                format: storage_texture_info.format,
                view_dimension: storage_texture_info.view_dimension,
            })
        }
        CustomBindingKind::BufferArray { .. } => Ok(wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        }),
        CustomBindingKind::TextureArray { .. } => {
            let sampled_texture_info =
                sampled_texture_info.unwrap_or(WgpuSampledTextureBindingInfo {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                });
            Ok(wgpu::BindingType::Texture {
                sample_type: sampled_texture_info.sample_type,
                view_dimension: sampled_texture_info.view_dimension,
                multisampled: sampled_texture_info.multisampled,
            })
        }
        CustomBindingKind::StorageTextureArray { .. } => {
            let Some(storage_texture_info) = storage_texture_info else {
                return Err(anyhow!(
                    "custom storage texture binding metadata is missing"
                ));
            };
            Ok(wgpu::BindingType::StorageTexture {
                access: storage_texture_info.access,
                format: storage_texture_info.format,
                view_dimension: storage_texture_info.view_dimension,
            })
        }
    }
}

fn binding_array_count(kind: CustomBindingKind) -> Option<NonZeroU32> {
    match kind {
        CustomBindingKind::BufferArray { count }
        | CustomBindingKind::TextureArray { count }
        | CustomBindingKind::StorageTextureArray { count } => NonZeroU32::new(count),
        _ => None,
    }
}

fn binding_kind_required_features(kind: CustomBindingKind) -> wgpu::Features {
    match kind {
        CustomBindingKind::BufferArray { .. } => {
            wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        }
        CustomBindingKind::TextureArray { .. } => wgpu::Features::TEXTURE_BINDING_ARRAY,
        CustomBindingKind::StorageTextureArray { .. } => {
            wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        }
        _ => wgpu::Features::empty(),
    }
}

fn map_texture_format(format: CustomTextureFormat) -> Option<wgpu::TextureFormat> {
    match format {
        CustomTextureFormat::R8Unorm => Some(wgpu::TextureFormat::R8Unorm),
        CustomTextureFormat::Rg8Unorm => Some(wgpu::TextureFormat::Rg8Unorm),
        CustomTextureFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8Unorm),
        CustomTextureFormat::Bgra8Unorm => Some(wgpu::TextureFormat::Bgra8Unorm),
        CustomTextureFormat::Rgba8UnormSrgb => Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        CustomTextureFormat::Bgra8UnormSrgb => Some(wgpu::TextureFormat::Bgra8UnormSrgb),
        CustomTextureFormat::Bc1Unorm => Some(wgpu::TextureFormat::Bc1RgbaUnorm),
        CustomTextureFormat::Bc1UnormSrgb => Some(wgpu::TextureFormat::Bc1RgbaUnormSrgb),
        CustomTextureFormat::Bc3Unorm => Some(wgpu::TextureFormat::Bc3RgbaUnorm),
        CustomTextureFormat::Bc3UnormSrgb => Some(wgpu::TextureFormat::Bc3RgbaUnormSrgb),
        CustomTextureFormat::Bc7Unorm => Some(wgpu::TextureFormat::Bc7RgbaUnorm),
        CustomTextureFormat::Bc7UnormSrgb => Some(wgpu::TextureFormat::Bc7RgbaUnormSrgb),
        CustomTextureFormat::Etc2Rgb8Unorm => Some(wgpu::TextureFormat::Etc2Rgb8Unorm),
        CustomTextureFormat::Etc2Rgb8UnormSrgb => Some(wgpu::TextureFormat::Etc2Rgb8UnormSrgb),
        CustomTextureFormat::Etc2Rgba8Unorm => Some(wgpu::TextureFormat::Etc2Rgba8Unorm),
        CustomTextureFormat::Etc2Rgba8UnormSrgb => Some(wgpu::TextureFormat::Etc2Rgba8UnormSrgb),
        CustomTextureFormat::Astc4x4Unorm => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::Unorm,
        }),
        CustomTextureFormat::Astc4x4UnormSrgb => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        CustomTextureFormat::Astc5x5Unorm => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::Unorm,
        }),
        CustomTextureFormat::Astc5x5UnormSrgb => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        CustomTextureFormat::Astc6x6Unorm => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::Unorm,
        }),
        CustomTextureFormat::Astc6x6UnormSrgb => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        CustomTextureFormat::Astc8x8Unorm => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::Unorm,
        }),
        CustomTextureFormat::Astc8x8UnormSrgb => Some(wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::UnormSrgb,
        }),
        CustomTextureFormat::PvrtcRgb2bppUnorm
        | CustomTextureFormat::PvrtcRgb2bppUnormSrgb
        | CustomTextureFormat::PvrtcRgba2bppUnorm
        | CustomTextureFormat::PvrtcRgba2bppUnormSrgb
        | CustomTextureFormat::PvrtcRgb4bppUnorm
        | CustomTextureFormat::PvrtcRgb4bppUnormSrgb
        | CustomTextureFormat::PvrtcRgba4bppUnorm
        | CustomTextureFormat::PvrtcRgba4bppUnormSrgb => None,
    }
}

fn texture_format_supported_by_device(
    device_features: wgpu::Features,
    texture_format: wgpu::TextureFormat,
) -> bool {
    device_features.contains(texture_format.required_features())
}

fn map_custom_storage_texture_format(format: CustomTextureFormat) -> Option<wgpu::TextureFormat> {
    match format {
        CustomTextureFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8Unorm),
        CustomTextureFormat::Bgra8Unorm => Some(wgpu::TextureFormat::Bgra8Unorm),
        _ => None,
    }
}

fn map_texture_view_dimension(
    dimension: CustomTextureDimension,
) -> Result<wgpu::TextureViewDimension> {
    match dimension {
        CustomTextureDimension::D2 => Ok(wgpu::TextureViewDimension::D2),
        CustomTextureDimension::D2Array { layers } => {
            if layers == 0 {
                return Err(anyhow!(
                    "custom texture array layers must be greater than zero"
                ));
            }
            Ok(wgpu::TextureViewDimension::D2Array)
        }
        CustomTextureDimension::Cube => Ok(wgpu::TextureViewDimension::Cube),
    }
}

fn map_depth_format(format: CustomDepthFormat) -> wgpu::TextureFormat {
    match format {
        CustomDepthFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
    }
}

fn map_depth_compare(compare: CustomDepthCompare) -> wgpu::CompareFunction {
    match compare {
        CustomDepthCompare::Always => wgpu::CompareFunction::Always,
        CustomDepthCompare::Less => wgpu::CompareFunction::Less,
        CustomDepthCompare::LessEqual => wgpu::CompareFunction::LessEqual,
        CustomDepthCompare::Greater => wgpu::CompareFunction::Greater,
        CustomDepthCompare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
    }
}

fn map_naga_storage_format(format: naga::StorageFormat) -> Result<wgpu::TextureFormat> {
    match format {
        naga::StorageFormat::R8Unorm => Ok(wgpu::TextureFormat::R8Unorm),
        naga::StorageFormat::R8Snorm => Ok(wgpu::TextureFormat::R8Snorm),
        naga::StorageFormat::R8Uint => Ok(wgpu::TextureFormat::R8Uint),
        naga::StorageFormat::R8Sint => Ok(wgpu::TextureFormat::R8Sint),
        naga::StorageFormat::R16Uint => Ok(wgpu::TextureFormat::R16Uint),
        naga::StorageFormat::R16Sint => Ok(wgpu::TextureFormat::R16Sint),
        naga::StorageFormat::R16Float => Ok(wgpu::TextureFormat::R16Float),
        naga::StorageFormat::Rg8Unorm => Ok(wgpu::TextureFormat::Rg8Unorm),
        naga::StorageFormat::Rg8Snorm => Ok(wgpu::TextureFormat::Rg8Snorm),
        naga::StorageFormat::Rg8Uint => Ok(wgpu::TextureFormat::Rg8Uint),
        naga::StorageFormat::Rg8Sint => Ok(wgpu::TextureFormat::Rg8Sint),
        naga::StorageFormat::R32Uint => Ok(wgpu::TextureFormat::R32Uint),
        naga::StorageFormat::R32Sint => Ok(wgpu::TextureFormat::R32Sint),
        naga::StorageFormat::R32Float => Ok(wgpu::TextureFormat::R32Float),
        naga::StorageFormat::Rg16Uint => Ok(wgpu::TextureFormat::Rg16Uint),
        naga::StorageFormat::Rg16Sint => Ok(wgpu::TextureFormat::Rg16Sint),
        naga::StorageFormat::Rg16Float => Ok(wgpu::TextureFormat::Rg16Float),
        naga::StorageFormat::Rgba8Unorm => Ok(wgpu::TextureFormat::Rgba8Unorm),
        naga::StorageFormat::Rgba8Snorm => Ok(wgpu::TextureFormat::Rgba8Snorm),
        naga::StorageFormat::Rgba8Uint => Ok(wgpu::TextureFormat::Rgba8Uint),
        naga::StorageFormat::Rgba8Sint => Ok(wgpu::TextureFormat::Rgba8Sint),
        naga::StorageFormat::Bgra8Unorm => Ok(wgpu::TextureFormat::Bgra8Unorm),
        naga::StorageFormat::Rgb10a2Uint => Ok(wgpu::TextureFormat::Rgb10a2Uint),
        naga::StorageFormat::Rgb10a2Unorm => Ok(wgpu::TextureFormat::Rgb10a2Unorm),
        naga::StorageFormat::Rg11b10Ufloat => Ok(wgpu::TextureFormat::Rg11b10Ufloat),
        naga::StorageFormat::Rg32Uint => Ok(wgpu::TextureFormat::Rg32Uint),
        naga::StorageFormat::Rg32Sint => Ok(wgpu::TextureFormat::Rg32Sint),
        naga::StorageFormat::Rg32Float => Ok(wgpu::TextureFormat::Rg32Float),
        naga::StorageFormat::Rgba16Uint => Ok(wgpu::TextureFormat::Rgba16Uint),
        naga::StorageFormat::Rgba16Sint => Ok(wgpu::TextureFormat::Rgba16Sint),
        naga::StorageFormat::Rgba16Float => Ok(wgpu::TextureFormat::Rgba16Float),
        naga::StorageFormat::Rgba32Uint => Ok(wgpu::TextureFormat::Rgba32Uint),
        naga::StorageFormat::Rgba32Sint => Ok(wgpu::TextureFormat::Rgba32Sint),
        naga::StorageFormat::Rgba32Float => Ok(wgpu::TextureFormat::Rgba32Float),
        naga::StorageFormat::R16Unorm => Ok(wgpu::TextureFormat::R16Unorm),
        naga::StorageFormat::R16Snorm => Ok(wgpu::TextureFormat::R16Snorm),
        naga::StorageFormat::Rg16Unorm => Ok(wgpu::TextureFormat::Rg16Unorm),
        naga::StorageFormat::Rg16Snorm => Ok(wgpu::TextureFormat::Rg16Snorm),
        naga::StorageFormat::Rgba16Unorm => Ok(wgpu::TextureFormat::Rgba16Unorm),
        naga::StorageFormat::Rgba16Snorm => Ok(wgpu::TextureFormat::Rgba16Snorm),
        unsupported => Err(anyhow!(
            "custom storage texture format {:?} is not supported on this wgpu renderer",
            unsupported
        )),
    }
}

fn map_naga_storage_access(access: naga::StorageAccess) -> Result<wgpu::StorageTextureAccess> {
    let has_load = access.contains(naga::StorageAccess::LOAD);
    let has_store = access.contains(naga::StorageAccess::STORE);

    match (has_load, has_store) {
        (true, true) => Ok(wgpu::StorageTextureAccess::ReadWrite),
        (true, false) => Ok(wgpu::StorageTextureAccess::ReadOnly),
        (false, true) => Ok(wgpu::StorageTextureAccess::WriteOnly),
        (false, false) => Err(anyhow!(
            "custom storage texture binding must allow load and/or store access"
        )),
    }
}

fn map_naga_storage_view_dimension(
    dimension: naga::ImageDimension,
    arrayed: bool,
) -> Result<wgpu::TextureViewDimension> {
    match (dimension, arrayed) {
        (naga::ImageDimension::D1, false) => Ok(wgpu::TextureViewDimension::D1),
        (naga::ImageDimension::D2, false) => Ok(wgpu::TextureViewDimension::D2),
        (naga::ImageDimension::D2, true) => Ok(wgpu::TextureViewDimension::D2Array),
        (naga::ImageDimension::D3, false) => Ok(wgpu::TextureViewDimension::D3),
        _ => Err(anyhow!(
            "custom storage texture dimension {:?} (arrayed={}) is not supported on wgpu",
            dimension,
            arrayed
        )),
    }
}

fn map_naga_sampled_view_dimension(
    dimension: naga::ImageDimension,
    arrayed: bool,
) -> Result<wgpu::TextureViewDimension> {
    match (dimension, arrayed) {
        (naga::ImageDimension::D1, false) => Ok(wgpu::TextureViewDimension::D1),
        (naga::ImageDimension::D2, false) => Ok(wgpu::TextureViewDimension::D2),
        (naga::ImageDimension::D2, true) => Ok(wgpu::TextureViewDimension::D2Array),
        (naga::ImageDimension::D3, false) => Ok(wgpu::TextureViewDimension::D3),
        (naga::ImageDimension::Cube, false) => Ok(wgpu::TextureViewDimension::Cube),
        (naga::ImageDimension::Cube, true) => Ok(wgpu::TextureViewDimension::CubeArray),
        _ => Err(anyhow!(
            "custom sampled texture dimension {:?} (arrayed={}) is not supported on wgpu",
            dimension,
            arrayed
        )),
    }
}

fn map_naga_sampled_texture_info(
    class: naga::ImageClass,
    dimension: naga::ImageDimension,
    arrayed: bool,
) -> Result<Option<WgpuSampledTextureBindingInfo>> {
    let (sample_type, multisampled) = match class {
        naga::ImageClass::Sampled { kind, multi } => {
            let sample_type = match kind {
                naga::ScalarKind::Float => wgpu::TextureSampleType::Float { filterable: true },
                naga::ScalarKind::Sint => wgpu::TextureSampleType::Sint,
                naga::ScalarKind::Uint => wgpu::TextureSampleType::Uint,
                naga::ScalarKind::Bool => {
                    return Err(anyhow!(
                        "custom sampled textures cannot use bool sample type"
                    ));
                }
                naga::ScalarKind::AbstractInt | naga::ScalarKind::AbstractFloat => {
                    return Err(anyhow!(
                        "custom sampled textures cannot use abstract sample types"
                    ));
                }
            };
            (sample_type, multi)
        }
        naga::ImageClass::Depth { multi } => (wgpu::TextureSampleType::Depth, multi),
        naga::ImageClass::External => {
            return Err(anyhow!(
                "custom external textures are not supported on this wgpu renderer"
            ));
        }
        naga::ImageClass::Storage { .. } => return Ok(None),
    };

    let view_dimension = map_naga_sampled_view_dimension(dimension, arrayed)?;

    Ok(Some(WgpuSampledTextureBindingInfo {
        sample_type,
        view_dimension,
        multisampled,
    }))
}

fn map_filter(filter: CustomFilterMode) -> wgpu::FilterMode {
    match filter {
        CustomFilterMode::Nearest => wgpu::FilterMode::Nearest,
        CustomFilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

fn map_mipmap_filter(filter: CustomFilterMode) -> wgpu::MipmapFilterMode {
    match filter {
        CustomFilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        CustomFilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

fn map_address_mode(address_mode: CustomAddressMode) -> wgpu::AddressMode {
    match address_mode {
        CustomAddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        CustomAddressMode::Repeat => wgpu::AddressMode::Repeat,
    }
}

fn map_vertex_format(format: CustomVertexFormat) -> Result<wgpu::VertexFormat> {
    match format {
        CustomVertexFormat::F32 => Ok(wgpu::VertexFormat::Float32),
        CustomVertexFormat::F32Vec2 => Ok(wgpu::VertexFormat::Float32x2),
        CustomVertexFormat::F32Vec3 => Ok(wgpu::VertexFormat::Float32x3),
        CustomVertexFormat::F32Vec4 => Ok(wgpu::VertexFormat::Float32x4),
        CustomVertexFormat::U32 => Ok(wgpu::VertexFormat::Uint32),
        CustomVertexFormat::U32Vec2 => Ok(wgpu::VertexFormat::Uint32x2),
        CustomVertexFormat::U32Vec3 => Ok(wgpu::VertexFormat::Uint32x3),
        CustomVertexFormat::U32Vec4 => Ok(wgpu::VertexFormat::Uint32x4),
        CustomVertexFormat::I32 => Ok(wgpu::VertexFormat::Sint32),
        CustomVertexFormat::I32Vec2 => Ok(wgpu::VertexFormat::Sint32x2),
        CustomVertexFormat::I32Vec3 => Ok(wgpu::VertexFormat::Sint32x3),
        CustomVertexFormat::I32Vec4 => Ok(wgpu::VertexFormat::Sint32x4),
    }
}

fn map_index_format(format: CustomIndexFormat) -> wgpu::IndexFormat {
    match format {
        CustomIndexFormat::U16 => wgpu::IndexFormat::Uint16,
        CustomIndexFormat::U32 => wgpu::IndexFormat::Uint32,
    }
}

fn map_primitive_topology(primitive: CustomPrimitiveTopology) -> wgpu::PrimitiveTopology {
    match primitive {
        CustomPrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        CustomPrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        CustomPrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        CustomPrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        CustomPrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

fn map_front_face(front_face: CustomFrontFace) -> wgpu::FrontFace {
    match front_face {
        CustomFrontFace::Ccw => wgpu::FrontFace::Ccw,
        CustomFrontFace::Cw => wgpu::FrontFace::Cw,
    }
}

fn map_cull_mode(cull_mode: CustomCullMode) -> Option<wgpu::Face> {
    match cull_mode {
        CustomCullMode::None => None,
        CustomCullMode::Front => Some(wgpu::Face::Front),
        CustomCullMode::Back => Some(wgpu::Face::Back),
    }
}

fn map_blend_state(blend_mode: gpui::CustomBlendMode) -> Option<wgpu::BlendState> {
    match blend_mode {
        gpui::CustomBlendMode::Default | gpui::CustomBlendMode::Alpha => {
            Some(wgpu::BlendState::ALPHA_BLENDING)
        }
        gpui::CustomBlendMode::Opaque => None,
        gpui::CustomBlendMode::PremultipliedAlpha => {
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)
        }
    }
}

fn depth_target_estimate_bytes(width: u32, height: u32, sample_count: u32) -> u64 {
    u64::from(width.max(1))
        .saturating_mul(u64::from(height.max(1)))
        .saturating_mul(4)
        .saturating_mul(u64::from(sample_count.max(1)))
}

fn clip_bounds_to_viewport(
    bounds: Bounds<ScaledPixels>,
    viewport_width: u32,
    viewport_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let min_x = bounds.origin.x.0.floor().max(0.0) as u32;
    let min_y = bounds.origin.y.0.floor().max(0.0) as u32;
    let max_x = (bounds.origin.x.0 + bounds.size.width.0)
        .ceil()
        .max(0.0)
        .min(viewport_width as f32) as u32;
    let max_y = (bounds.origin.y.0 + bounds.size.height.0)
        .ceil()
        .max(0.0)
        .min(viewport_height as f32) as u32;

    if max_x <= min_x || max_y <= min_y {
        return None;
    }

    Some((min_x, min_y, max_x - min_x, max_y - min_y))
}

fn build_attribute_locations(
    vertex_fetches: &[CustomVertexFetch],
) -> Result<HashMap<&'static str, u32>> {
    let mut locations = HashMap::new();
    let mut used_locations = BTreeSet::new();

    for fetch in vertex_fetches {
        for attribute in &fetch.layout.attributes {
            if let Some(location) = attribute.location {
                if !used_locations.insert(location) {
                    return Err(anyhow!(
                        "custom draw vertex attribute locations must be unique (duplicate {})",
                        location
                    ));
                }
                locations.insert(attribute.name.as_str(), location);
            }
        }
    }

    let mut next_location = 0u32;
    for fetch in vertex_fetches {
        for attribute in &fetch.layout.attributes {
            let attribute_name = attribute.name.as_str();
            if locations.contains_key(attribute_name) {
                continue;
            }
            while used_locations.contains(&next_location) {
                next_location += 1;
            }
            locations.insert(attribute_name, next_location);
            used_locations.insert(next_location);
            next_location += 1;
        }
    }

    Ok(locations)
}

fn push_constants_slot(bindings: &[CustomBindingDesc]) -> CustomBindingSlot {
    let mut max_group = 0u32;
    for binding in bindings {
        let slot = binding.slot.unwrap_or(CustomBindingSlot {
            group: 0,
            binding: binding.name.index(),
        });
        max_group = max_group.max(slot.group);
    }

    CustomBindingSlot {
        group: max_group.saturating_add(1),
        binding: 0,
    }
}

fn apply_push_constants(
    module: &mut naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_indices: &[usize],
    push_constants: Option<CustomPushConstantsDesc>,
    slot: CustomBindingSlot,
) -> Result<Option<PushConstantsInfo>> {
    let mut push_constant_handle = None;
    for (handle, variable) in module.global_variables.iter() {
        if variable.space != naga::AddressSpace::Immediate {
            continue;
        }

        let used = entry_indices.iter().any(|entry_index| {
            let entry_info = info.get_entry_point(*entry_index);
            !entry_info[handle].is_empty()
        });
        if !used {
            continue;
        }

        if push_constant_handle.is_some() {
            return Err(anyhow!(
                "custom draw shaders may declare at most one push constants block"
            ));
        }
        push_constant_handle = Some(handle);
    }

    let Some(handle) = push_constant_handle else {
        if push_constants.is_some() {
            return Err(anyhow!(
                "push constants were provided but the shader has no push constant block"
            ));
        }
        return Ok(None);
    };

    let push_constants = push_constants
        .ok_or_else(|| anyhow!("shader declares push constants but none were provided"))?;

    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|error| anyhow!("push constants layout failed: {error}"))?;
    let layout = &layouter[module.global_variables[handle].ty];

    if layout.size != push_constants.size {
        return Err(anyhow!(
            "push constants size mismatch (expected {}, shader reports {})",
            push_constants.size,
            layout.size
        ));
    }

    let variable = module.global_variables.get_mut(handle);
    variable.space = naga::AddressSpace::Uniform;
    variable.binding = None;
    if variable.name.is_none() {
        variable.name = Some("push_constants".to_string());
    }

    let name = variable
        .name
        .clone()
        .unwrap_or_else(|| "push_constants".to_string());

    Ok(Some(PushConstantsInfo {
        name: Box::leak(name.into_boxed_str()),
        size: push_constants.size,
        slot,
    }))
}

fn build_binding_maps(
    bindings: &[CustomBindingDesc],
) -> (
    HashMap<&'static str, BindingInfo>,
    HashMap<(u32, u32), BindingInfo>,
) {
    let mut by_name = HashMap::new();
    let mut by_slot = HashMap::new();

    for binding in bindings {
        let slot = binding.slot.unwrap_or(CustomBindingSlot {
            group: 0,
            binding: binding.name.index(),
        });
        let info = BindingInfo {
            kind: binding.kind,
            slot,
        };
        by_name.insert(binding.name.as_str(), info);
        by_slot.insert((slot.group, slot.binding), info);
    }

    (by_name, by_slot)
}

fn collect_storage_texture_binding_info(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_point_index: usize,
) -> Result<HashMap<(u32, u32), WgpuStorageTextureBindingInfo>> {
    let entry_info = info.get_entry_point(entry_point_index);
    let mut storage_texture_infos = HashMap::new();

    for (handle, variable) in module.global_variables.iter() {
        if entry_info[handle].is_empty() {
            continue;
        }

        let Some(binding) = variable.binding else {
            continue;
        };

        let image = match module.types[variable.ty].inner {
            naga::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => Some((dim, arrayed, class)),
            naga::TypeInner::BindingArray { base, .. } => match module.types[base].inner {
                naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => Some((dim, arrayed, class)),
                _ => None,
            },
            _ => None,
        };

        let Some((dim, arrayed, naga::ImageClass::Storage { format, access })) = image else {
            continue;
        };

        let format = map_naga_storage_format(format)?;
        let view_dimension = map_naga_storage_view_dimension(dim, arrayed)?;
        let access = map_naga_storage_access(access)?;

        storage_texture_infos.insert(
            (binding.group, binding.binding),
            WgpuStorageTextureBindingInfo {
                format,
                access,
                view_dimension,
            },
        );
    }

    Ok(storage_texture_infos)
}

fn collect_sampled_texture_binding_info(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_point_index: usize,
) -> Result<HashMap<(u32, u32), WgpuSampledTextureBindingInfo>> {
    let entry_info = info.get_entry_point(entry_point_index);
    let mut sampled_texture_infos = HashMap::new();

    for (handle, variable) in module.global_variables.iter() {
        if entry_info[handle].is_empty() {
            continue;
        }

        let Some(binding) = variable.binding else {
            continue;
        };

        let image = match module.types[variable.ty].inner {
            naga::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => Some((dim, arrayed, class)),
            naga::TypeInner::BindingArray { base, .. } => match module.types[base].inner {
                naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => Some((dim, arrayed, class)),
                _ => None,
            },
            _ => None,
        };

        let Some((dim, arrayed, class)) = image else {
            continue;
        };

        let Some(sampled_texture_info) = map_naga_sampled_texture_info(class, dim, arrayed)? else {
            continue;
        };

        sampled_texture_infos.insert((binding.group, binding.binding), sampled_texture_info);
    }

    Ok(sampled_texture_infos)
}

fn merge_storage_texture_binding_infos(
    target: &mut HashMap<(u32, u32), WgpuStorageTextureBindingInfo>,
    source: HashMap<(u32, u32), WgpuStorageTextureBindingInfo>,
) -> Result<()> {
    for (slot, info) in source {
        if let Some(existing) = target.get(&slot) {
            if existing != &info {
                return Err(anyhow!(
                    "custom storage texture binding metadata mismatch at @group({}) @binding({})",
                    slot.0,
                    slot.1
                ));
            }
            continue;
        }
        target.insert(slot, info);
    }

    Ok(())
}

fn merge_sampled_texture_binding_infos(
    target: &mut HashMap<(u32, u32), WgpuSampledTextureBindingInfo>,
    source: HashMap<(u32, u32), WgpuSampledTextureBindingInfo>,
) -> Result<()> {
    for (slot, info) in source {
        if let Some(existing) = target.get(&slot) {
            if existing != &info {
                return Err(anyhow!(
                    "custom sampled texture binding metadata mismatch at @group({}) @binding({})",
                    slot.0,
                    slot.1
                ));
            }
            continue;
        }
        target.insert(slot, info);
    }

    Ok(())
}

fn assign_vertex_locations(
    module: &mut naga::Module,
    vertex_entry_index: usize,
    attribute_locations: &HashMap<&'static str, u32>,
) -> Result<()> {
    for (entry_index, entry_point) in module.entry_points.iter().enumerate() {
        if entry_point.stage != naga::ShaderStage::Vertex {
            continue;
        }

        for argument in entry_point.function.arguments.iter() {
            if argument.binding.is_some() {
                continue;
            }

            let mut ty = module.types[argument.ty].clone();
            let members = match ty.inner {
                naga::TypeInner::Struct {
                    ref mut members, ..
                } => members,
                _ => {
                    return Err(anyhow!(
                        "vertex entry '{}' input is not a struct",
                        entry_point.name
                    ));
                }
            };

            let mut modified = false;

            if entry_index == vertex_entry_index {
                for member in members.iter_mut() {
                    if member.binding.is_some() {
                        continue;
                    }
                    let Some(member_name) = member.name.as_deref() else {
                        return Err(anyhow!("vertex input member is missing a name"));
                    };
                    let Some(location) = attribute_locations.get(member_name) else {
                        return Err(anyhow!(
                            "vertex input '{}' was not provided in the custom vertex layout",
                            member_name
                        ));
                    };
                    member.binding = Some(naga::Binding::Location {
                        location: *location,
                        interpolation: None,
                        sampling: None,
                        blend_src: None,
                        per_primitive: false,
                    });
                    modified = true;
                }
            } else {
                let mut location = 0u32;
                for member in members.iter_mut() {
                    if member.binding.is_none() {
                        member.binding = Some(naga::Binding::Location {
                            location,
                            interpolation: None,
                            sampling: None,
                            blend_src: None,
                            per_primitive: false,
                        });
                        location = location.saturating_add(1);
                        modified = true;
                    }
                }
            }

            if modified {
                module.types.replace(argument.ty, ty);
            }
        }
    }

    Ok(())
}

fn assign_resource_bindings(
    module: &mut naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_point_name: &str,
    entry_point_index: usize,
    bindings_by_name: &HashMap<&'static str, BindingInfo>,
    bindings_by_slot: &HashMap<(u32, u32), BindingInfo>,
) -> Result<()> {
    let entry_point_info = info.get_entry_point(entry_point_index);
    let mut updates = Vec::new();

    for (handle, variable) in module.global_variables.iter() {
        if entry_point_info[handle].is_empty() {
            continue;
        }

        match variable.space {
            naga::AddressSpace::Storage { .. }
            | naga::AddressSpace::Uniform
            | naga::AddressSpace::Handle => {}
            _ => continue,
        }

        let variable_name = variable.name.as_deref().unwrap_or("<unnamed>");

        let binding_info = if let Some(binding) = variable.binding {
            *bindings_by_slot
                .get(&(binding.group, binding.binding))
                .ok_or_else(|| {
                    anyhow!(
                        "explicit binding @group({}) @binding({}) is not declared for '{}'",
                        binding.group,
                        binding.binding,
                        variable_name
                    )
                })?
        } else {
            *bindings_by_name.get(variable_name).ok_or_else(|| {
                anyhow!(
                    "custom draw binding '{}' is not declared in the pipeline descriptor",
                    variable_name
                )
            })?
        };

        validate_binding_kind(
            module,
            variable,
            binding_info.kind,
            entry_point_name,
            variable_name,
        )?;

        if variable.binding.is_none() {
            updates.push((handle, binding_info.slot));
        }
    }

    for (handle, slot) in updates {
        let variable = module.global_variables.get_mut(handle);
        variable.binding = Some(naga::ResourceBinding {
            group: slot.group,
            binding: slot.binding,
        });
    }

    Ok(())
}

fn validate_binding_kind(
    module: &naga::Module,
    variable: &naga::GlobalVariable,
    binding_kind: CustomBindingKind,
    entry_point_name: &str,
    variable_name: &str,
) -> Result<()> {
    match binding_kind {
        CustomBindingKind::BufferArray { count } => match module.types[variable.ty].inner {
            naga::TypeInner::BindingArray { size, .. } => {
                validate_binding_array_size(size, count, entry_point_name, variable_name)?;
                match variable.space {
                    naga::AddressSpace::Storage { .. } => Ok(()),
                    _ => Err(anyhow!(
                        "binding '{}' in entry '{}' must be a storage buffer array",
                        variable_name,
                        entry_point_name
                    )),
                }
            }
            _ => Err(anyhow!(
                "binding '{}' in entry '{}' must be a storage buffer array",
                variable_name,
                entry_point_name
            )),
        },
        CustomBindingKind::TextureArray { count } => match module.types[variable.ty].inner {
            naga::TypeInner::BindingArray { base, size } => match module.types[base].inner {
                naga::TypeInner::Image {
                    class: naga::ImageClass::Sampled { .. },
                    ..
                } => {
                    validate_binding_array_size(size, count, entry_point_name, variable_name)?;
                    Ok(())
                }
                _ => Err(anyhow!(
                    "binding '{}' in entry '{}' must be a sampled texture array",
                    variable_name,
                    entry_point_name
                )),
            },
            _ => Err(anyhow!(
                "binding '{}' in entry '{}' must be a sampled texture array",
                variable_name,
                entry_point_name
            )),
        },
        CustomBindingKind::StorageTextureArray { count } => match module.types[variable.ty].inner {
            naga::TypeInner::BindingArray { base, size } => match module.types[base].inner {
                naga::TypeInner::Image {
                    class: naga::ImageClass::Storage { .. },
                    ..
                } => {
                    validate_binding_array_size(size, count, entry_point_name, variable_name)?;
                    Ok(())
                }
                _ => Err(anyhow!(
                    "binding '{}' in entry '{}' must be a storage texture array",
                    variable_name,
                    entry_point_name
                )),
            },
            _ => Err(anyhow!(
                "binding '{}' in entry '{}' must be a storage texture array",
                variable_name,
                entry_point_name
            )),
        },
        CustomBindingKind::Texture => match module.types[variable.ty].inner {
            naga::TypeInner::Image {
                class: naga::ImageClass::Sampled { .. },
                ..
            } => Ok(()),
            _ => Err(anyhow!(
                "binding '{}' in entry '{}' must be a sampled texture",
                variable_name,
                entry_point_name
            )),
        },
        CustomBindingKind::Sampler => match module.types[variable.ty].inner {
            naga::TypeInner::Sampler { .. } => Ok(()),
            _ => Err(anyhow!(
                "binding '{}' in entry '{}' must be a sampler",
                variable_name,
                entry_point_name
            )),
        },
        CustomBindingKind::Uniform { size } => {
            if variable.space != naga::AddressSpace::Uniform {
                return Err(anyhow!(
                    "binding '{}' in entry '{}' must be a uniform buffer",
                    variable_name,
                    entry_point_name
                ));
            }

            let mut layouter = naga::proc::Layouter::default();
            layouter
                .update(module.to_ctx())
                .map_err(|error| anyhow!("uniform layout failed: {error}"))?;
            let layout = &layouter[variable.ty];
            if layout.size != size {
                return Err(anyhow!(
                    "binding '{}' size mismatch (expected {}, shader reports {})",
                    variable_name,
                    size,
                    layout.size
                ));
            }

            Ok(())
        }
        CustomBindingKind::Buffer => match variable.space {
            naga::AddressSpace::Storage { .. } => Ok(()),
            _ => Err(anyhow!(
                "binding '{}' in entry '{}' must be a storage buffer",
                variable_name,
                entry_point_name
            )),
        },
        CustomBindingKind::StorageTexture => match module.types[variable.ty].inner {
            naga::TypeInner::Image {
                class: naga::ImageClass::Storage { .. },
                ..
            } => Ok(()),
            _ => Err(anyhow!(
                "binding '{}' in entry '{}' must be a storage texture",
                variable_name,
                entry_point_name
            )),
        },
    }
}

fn validate_binding_array_size(
    size: naga::ArraySize,
    expected: u32,
    entry_point_name: &str,
    variable_name: &str,
) -> Result<()> {
    let actual = match size {
        naga::ArraySize::Constant(size) => size.get(),
        naga::ArraySize::Pending(_) => {
            return Err(anyhow!(
                "binding '{}' in entry '{}' must use a constant binding array length",
                variable_name,
                entry_point_name
            ));
        }
        naga::ArraySize::Dynamic => {
            return Err(anyhow!(
                "binding '{}' in entry '{}' must not use a runtime-sized binding array",
                variable_name,
                entry_point_name
            ));
        }
    };

    if actual != expected {
        return Err(anyhow!(
            "binding '{}' array length mismatch (expected {}, shader reports {})",
            variable_name,
            expected,
            actual
        ));
    }

    Ok(())
}
