use anyhow::anyhow;
use gpui::{
    Bounds, CustomAddressMode, CustomBindingDesc, CustomBindingKind, CustomBindingSlot,
    CustomBindingValue, CustomBufferDesc, CustomBufferId, CustomBufferSource, CustomCompute,
    CustomComputePipelineDesc, CustomComputePipelineId, CustomCullMode, CustomDepthTargetDesc,
    CustomDepthTargetId, CustomDraw, CustomDrawRegistry, CustomDrawResourceStats, CustomFilterMode,
    CustomFrameDiagnostics, CustomFrontFace, CustomGpuFrameProfile, CustomIndexBuffer,
    CustomIndexFormat, CustomPipelineDesc, CustomPipelineId, CustomPrimitiveTopology,
    CustomRenderTargetDesc, CustomSamplerDesc, CustomSamplerId, CustomTextureBufferUpdate,
    CustomTextureDesc, CustomTextureDimension, CustomTextureFormat, CustomTextureId,
    CustomTextureUpdate, CustomTextureUsage, CustomVertexFetch, CustomVertexFormat, Result,
    ScaledPixels,
};
use parking_lot::Mutex;
use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
struct WgpuBindingSpec {
    kind: CustomBindingKind,
    slot: CustomBindingSlot,
}

#[derive(Clone)]
struct WgpuCustomPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bindings: Vec<WgpuBindingSpec>,
    vertex_fetch_count: usize,
}

#[derive(Clone)]
struct WgpuCustomComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
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
    width: u32,
    height: u32,
    mip_level_count: u32,
    format: CustomTextureFormat,
    is_render_target: bool,
}

#[derive(Clone, Copy)]
struct BindingInfo {
    kind: CustomBindingKind,
    slot: CustomBindingSlot,
}

#[derive(Clone, Copy)]
struct WgpuStorageTextureBindingInfo {
    format: wgpu::TextureFormat,
    access: wgpu::StorageTextureAccess,
    view_dimension: wgpu::TextureViewDimension,
}

enum OwnedBindingResource {
    Buffer {
        binding: u32,
        buffer: wgpu::Buffer,
        offset: u64,
        size: Option<NonZeroU64>,
    },
    Texture {
        binding: u32,
        view: wgpu::TextureView,
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
    samplers: Mutex<Vec<Option<wgpu::Sampler>>>,
}

impl WgpuCustomDrawRegistry {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            surface_format,
            pipelines: Mutex::new(Vec::new()),
            compute_pipelines: Mutex::new(Vec::new()),
            buffers: Mutex::new(Vec::new()),
            textures: Mutex::new(Vec::new()),
            samplers: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn draw_window_custom_draws(
        &self,
        draws: &[CustomDraw],
        encoder: &mut wgpu::CommandEncoder,
        frame_view: &wgpu::TextureView,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        if draws.is_empty() {
            return;
        }

        let pipelines = self.pipelines.lock().clone();
        let buffers = self.buffers.lock().clone();
        let textures = self.textures.lock().clone();
        let samplers = self.samplers.lock().clone();

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
            depth_stencil_attachment: None,
            ..Default::default()
        });

        let mut temporary_buffers: Vec<wgpu::Buffer> = Vec::new();

        for draw in draws {
            if draw.target.is_some() {
                log::warn!("custom draw offscreen targets are not yet implemented on wgpu");
                continue;
            }

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

            let Some((scissor_x, scissor_y, scissor_width, scissor_height)) =
                clip_bounds_to_viewport(draw.content_mask.bounds, viewport_width, viewport_height)
            else {
                continue;
            };

            pass.set_pipeline(&pipeline.pipeline);
            pass.set_scissor_rect(scissor_x, scissor_y, scissor_width, scissor_height);

            let mut draw_failed = false;

            for (slot_index, vertex_buffer) in draw.vertex_buffers.iter().enumerate() {
                if let Err(error) = self.set_vertex_buffer(
                    &mut pass,
                    slot_index as u32,
                    &vertex_buffer.source,
                    &buffers,
                    &mut temporary_buffers,
                ) {
                    log::warn!("custom draw vertex buffer binding failed: {error}");
                    draw_failed = true;
                    break;
                }
            }

            if draw_failed {
                continue;
            }

            let bind_group = match self.create_draw_bind_group(
                pipeline,
                &draw.bindings,
                &buffers,
                &textures,
                &samplers,
                &mut temporary_buffers,
            ) {
                Ok(bind_group) => bind_group,
                Err(error) => {
                    log::warn!("custom draw bind group creation failed: {error}");
                    continue;
                }
            };

            pass.set_bind_group(0, &bind_group, &[]);

            if let Some(index_buffer) = &draw.index_buffer {
                if let Err(error) =
                    self.set_index_buffer(&mut pass, index_buffer, &buffers, &mut temporary_buffers)
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

    pub(crate) fn dispatch_custom_computes(
        &self,
        computes: &[CustomCompute],
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if computes.is_empty() {
            return;
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

            let bind_group = match self.create_compute_bind_group(
                pipeline,
                &compute.bindings,
                &buffers,
                &textures,
                &samplers,
                &mut temporary_buffers,
            ) {
                Ok(bind_group) => bind_group,
                Err(error) => {
                    log::warn!("custom compute bind group creation failed: {error}");
                    continue;
                }
            };

            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                compute.workgroup_count[0],
                compute.workgroup_count[1],
                compute.workgroup_count[2],
            );
        }
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

    fn create_draw_bind_group(
        &self,
        pipeline: &WgpuCustomPipeline,
        binding_values: &[CustomBindingValue],
        buffers: &[Option<WgpuCustomBuffer>],
        textures: &[Option<WgpuCustomTexture>],
        samplers: &[Option<wgpu::Sampler>],
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<wgpu::BindGroup> {
        let mut owned_resources = Vec::with_capacity(binding_values.len());

        for (binding_spec, binding_value) in pipeline.bindings.iter().zip(binding_values.iter()) {
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
                (CustomBindingKind::Texture, CustomBindingValue::Texture(id)) => {
                    let Some(Some(texture_entry)) = textures.get(id.0 as usize) else {
                        return Err(anyhow!("custom draw texture {} is missing", id.0));
                    };
                    owned_resources.push(OwnedBindingResource::Texture {
                        binding: binding_spec.slot.binding,
                        view: texture_entry.view.clone(),
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
                OwnedBindingResource::Texture { binding, view } => {
                    bind_group_entries.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::TextureView(view),
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

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom_draw_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &bind_group_entries,
        }))
    }

    fn create_compute_bind_group(
        &self,
        pipeline: &WgpuCustomComputePipeline,
        binding_values: &[CustomBindingValue],
        buffers: &[Option<WgpuCustomBuffer>],
        textures: &[Option<WgpuCustomTexture>],
        samplers: &[Option<wgpu::Sampler>],
        temporary_buffers: &mut Vec<wgpu::Buffer>,
    ) -> Result<wgpu::BindGroup> {
        let mut owned_resources = Vec::with_capacity(binding_values.len());

        for (binding_spec, binding_value) in pipeline.bindings.iter().zip(binding_values.iter()) {
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
                OwnedBindingResource::Texture { binding, view } => {
                    bind_group_entries.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::TextureView(view),
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

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom_compute_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &bind_group_entries,
        }))
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
            mapped_range[..copy_len].copy_from_slice(&data[..copy_len]);
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
            mapped_range[..copy_len].copy_from_slice(&data[..copy_len]);
        }

        buffer.unmap();

        WgpuCustomBuffer {
            buffer,
            size: buffer_size,
        }
    }

    fn create_render_pipeline(&self, desc: CustomPipelineDesc) -> Result<WgpuCustomPipeline> {
        if desc.push_constants.is_some() {
            return Err(anyhow!(
                "custom draw push constants are not yet supported on wgpu"
            ));
        }
        if !desc.color_targets.is_empty() {
            return Err(anyhow!(
                "custom draw offscreen color targets are not yet supported on wgpu"
            ));
        }
        if desc.state.depth.is_some() {
            return Err(anyhow!(
                "custom draw depth state is not yet supported on wgpu"
            ));
        }
        if desc.state.sample_count != 1 {
            return Err(anyhow!(
                "custom draw sample counts above 1 are not yet supported on wgpu"
            ));
        }

        for binding in &desc.bindings {
            let slot = binding.slot.unwrap_or(CustomBindingSlot {
                group: 0,
                binding: binding.name.index(),
            });
            if slot.group != 0 {
                return Err(anyhow!(
                    "custom draw bind groups above group 0 are not yet supported on wgpu"
                ));
            }
            match binding.kind {
                CustomBindingKind::Buffer
                | CustomBindingKind::Texture
                | CustomBindingKind::Sampler
                | CustomBindingKind::Uniform { .. } => {}
                CustomBindingKind::BufferArray { .. }
                | CustomBindingKind::TextureArray { .. }
                | CustomBindingKind::StorageTextureArray { .. } => {
                    return Err(anyhow!(
                        "custom draw binding arrays are not yet supported on wgpu"
                    ));
                }
                CustomBindingKind::StorageTexture => {
                    return Err(anyhow!(
                        "custom draw storage textures are not yet supported on wgpu"
                    ));
                }
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

        let attribute_locations = build_attribute_locations(&desc.vertex_fetches)?;
        assign_vertex_locations(&mut module, vertex_entry_index, &attribute_locations)?;

        let (bindings_by_name, bindings_by_slot) = build_binding_maps(&desc.bindings);
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

        let rewritten_wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .map_err(|error| anyhow!("custom draw WGSL rewrite failed: {error}"))?;

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("custom_draw_shader"),
                source: wgpu::ShaderSource::Wgsl(rewritten_wgsl.into()),
            });

        let mut binding_specs = Vec::with_capacity(desc.bindings.len());
        let mut bind_group_layout_entries = Vec::with_capacity(desc.bindings.len());

        for binding in &desc.bindings {
            let slot = binding.slot.unwrap_or(CustomBindingSlot {
                group: 0,
                binding: binding.name.index(),
            });
            binding_specs.push(WgpuBindingSpec {
                kind: binding.kind,
                slot,
            });
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: slot.binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: map_binding_type(binding.kind, None)?,
                count: None,
            });
        }

        bind_group_layout_entries.sort_by_key(|entry| entry.binding);

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("custom_draw_bind_group_layout"),
                    entries: &bind_group_layout_entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("custom_draw_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
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

        let color_target = Some(wgpu::ColorTargetState {
            format: self.surface_format,
            blend: map_blend_state(desc.state.blend),
            write_mask: wgpu::ColorWrites::ALL,
        });

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
                    targets: &[color_target],
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        Ok(WgpuCustomPipeline {
            pipeline: render_pipeline,
            bind_group_layout,
            bindings: binding_specs,
            vertex_fetch_count: desc.vertex_fetches.len(),
        })
    }

    fn create_compute_pipeline_impl(
        &self,
        desc: CustomComputePipelineDesc,
    ) -> Result<WgpuCustomComputePipeline> {
        if desc.push_constants.is_some() {
            return Err(anyhow!(
                "custom compute push constants are not yet supported on wgpu"
            ));
        }

        for binding in &desc.bindings {
            let slot = binding.slot.unwrap_or(CustomBindingSlot {
                group: 0,
                binding: binding.name.index(),
            });
            if slot.group != 0 {
                return Err(anyhow!(
                    "custom compute bind groups above group 0 are not yet supported on wgpu"
                ));
            }
            match binding.kind {
                CustomBindingKind::Buffer
                | CustomBindingKind::Texture
                | CustomBindingKind::Sampler
                | CustomBindingKind::Uniform { .. }
                | CustomBindingKind::StorageTexture => {}
                CustomBindingKind::BufferArray { .. }
                | CustomBindingKind::TextureArray { .. }
                | CustomBindingKind::StorageTextureArray { .. } => {
                    return Err(anyhow!(
                        "custom compute binding arrays are not yet supported on wgpu"
                    ));
                }
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

        let (bindings_by_name, bindings_by_slot) = build_binding_maps(&desc.bindings);
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

        let rewritten_wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .map_err(|error| anyhow!("custom compute WGSL rewrite failed: {error}"))?;

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("custom_compute_shader"),
                source: wgpu::ShaderSource::Wgsl(rewritten_wgsl.into()),
            });

        let mut binding_specs = Vec::with_capacity(desc.bindings.len());
        let mut bind_group_layout_entries = Vec::with_capacity(desc.bindings.len());

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
            )?;

            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: slot.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: binding_type,
                count: None,
            });
        }

        bind_group_layout_entries.sort_by_key(|entry| entry.binding);

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("custom_compute_bind_group_layout"),
                    entries: &bind_group_layout_entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("custom_compute_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
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
            bind_group_layout,
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
                    .saturating_mul(u64::from(block.bytes)),
            );
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

        if texture.format.is_compressed() {
            return Err(anyhow!(
                "compressed custom textures are not yet supported on wgpu"
            ));
        }

        let block = texture.format.block_info();
        let mip_width = (texture.width >> level).max(1);
        let mip_height = (texture.height >> level).max(1);
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

        let required_bytes = u64::from(upload_bytes_per_row).saturating_mul(u64::from(mip_height));
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
                rows_per_image: Some(mip_height),
            },
            wgpu::Extent3d {
                width: mip_width,
                height: mip_height,
                depth_or_array_layers: 1,
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

    fn set_gpu_profiling_enabled(&self, _enabled: bool) -> Result<()> {
        Err(anyhow!(
            "custom draw GPU profiling is not yet supported on wgpu"
        ))
    }

    fn take_last_gpu_profile(&self) -> Option<CustomGpuFrameProfile> {
        None
    }

    fn set_frame_diagnostics_enabled(&self, _enabled: bool) -> Result<()> {
        Err(anyhow!(
            "custom draw frame diagnostics are not yet supported on wgpu"
        ))
    }

    fn take_last_frame_diagnostics(&self) -> Option<CustomFrameDiagnostics> {
        None
    }

    fn resource_stats(&self) -> CustomDrawResourceStats {
        let pipelines = self.pipelines.lock();
        let compute_pipelines = self.compute_pipelines.lock();
        let buffers = self.buffers.lock();
        let textures = self.textures.lock();
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
            depth_target_count: 0,
            depth_target_bytes: 0,
            sampler_count: samplers.iter().filter(|entry| entry.is_some()).count() as u32,
        }
    }

    fn texture_format_supported(&self, format: CustomTextureFormat) -> bool {
        map_texture_format(format).is_some()
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
        if desc.dimension != CustomTextureDimension::D2 {
            return Err(anyhow!(
                "custom texture arrays and cubemaps are not yet supported on wgpu"
            ));
        }
        if desc.format.is_compressed() {
            return Err(anyhow!(
                "compressed custom textures are not yet supported on wgpu"
            ));
        }

        let Some(texture_format) = map_texture_format(desc.format) else {
            return Err(anyhow!(
                "custom texture format {:?} is not supported by this wgpu renderer",
                desc.format
            ));
        };

        let mut texture_usage = wgpu::TextureUsages::COPY_DST;
        if desc.usage.contains(CustomTextureUsage::SAMPLED) {
            texture_usage |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        if desc.usage.contains(CustomTextureUsage::STORAGE) {
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
            texture_usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        if texture_usage == wgpu::TextureUsages::COPY_DST {
            return Err(anyhow!(
                "custom texture usage must include sampled and/or storage usage"
            ));
        }

        let mip_level_count = desc.data.len().max(1) as u32;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&desc.name),
            size: wgpu::Extent3d {
                width: desc.width.max(1),
                height: desc.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: texture_usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_entry = WgpuCustomTexture {
            texture,
            view,
            width: desc.width.max(1),
            height: desc.height.max(1),
            mip_level_count,
            format: desc.format,
            is_render_target: false,
        };

        for (level, data) in desc.data.iter().enumerate() {
            self.upload_texture_level(&texture_entry, level as u32, data, None)?;
        }

        let mut textures = self.textures.lock();
        let texture_id = alloc_slot(&mut textures, texture_entry);
        Ok(CustomTextureId(texture_id))
    }

    fn create_render_target(&self, _desc: CustomRenderTargetDesc) -> Result<CustomTextureId> {
        Err(anyhow!(
            "custom draw offscreen render targets are not yet supported on wgpu"
        ))
    }

    fn update_texture(&self, id: CustomTextureId, update: CustomTextureUpdate) -> Result<()> {
        let textures = self.textures.lock();
        let Some(Some(texture_entry)) = textures.get(id.0 as usize) else {
            return Err(anyhow!("custom draw texture {} not found", id.0));
        };
        self.upload_texture_level(
            texture_entry,
            update.level,
            &update.data,
            update.bytes_per_row,
        )
    }

    fn update_texture_from_buffer(
        &self,
        _id: CustomTextureId,
        _update: CustomTextureBufferUpdate,
    ) -> Result<()> {
        Err(anyhow!(
            "custom texture updates from buffer are not yet supported on wgpu"
        ))
    }

    fn remove_texture(&self, id: CustomTextureId) {
        let mut textures = self.textures.lock();
        if let Some(slot) = textures.get_mut(id.0 as usize) {
            slot.take();
        }
    }

    fn create_depth_target(&self, _desc: CustomDepthTargetDesc) -> Result<CustomDepthTargetId> {
        Err(anyhow!(
            "custom draw depth targets are not yet supported on wgpu"
        ))
    }

    fn remove_depth_target(&self, _id: CustomDepthTargetId) {}

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
        CustomBindingKind::Texture => Ok(wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        }),
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
        CustomBindingKind::BufferArray { .. }
        | CustomBindingKind::TextureArray { .. }
        | CustomBindingKind::StorageTextureArray { .. } => Err(anyhow!(
            "custom draw binding arrays are not yet supported on wgpu"
        )),
    }
}

fn map_texture_format(format: CustomTextureFormat) -> Option<wgpu::TextureFormat> {
    match format {
        CustomTextureFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8Unorm),
        CustomTextureFormat::Bgra8Unorm => Some(wgpu::TextureFormat::Bgra8Unorm),
        CustomTextureFormat::Rgba8UnormSrgb => Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        CustomTextureFormat::Bgra8UnormSrgb => Some(wgpu::TextureFormat::Bgra8UnormSrgb),
        _ => None,
    }
}

fn map_custom_storage_texture_format(format: CustomTextureFormat) -> Option<wgpu::TextureFormat> {
    match format {
        CustomTextureFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8Unorm),
        CustomTextureFormat::Bgra8Unorm => Some(wgpu::TextureFormat::Bgra8Unorm),
        _ => None,
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

        let naga::TypeInner::Image {
            dim,
            arrayed,
            class: naga::ImageClass::Storage { format, access },
        } = module.types[variable.ty].inner
        else {
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
        CustomBindingKind::BufferArray { .. }
        | CustomBindingKind::TextureArray { .. }
        | CustomBindingKind::StorageTextureArray { .. } => Err(anyhow!(
            "binding '{}' in entry '{}' uses binding arrays, which are not yet supported on wgpu",
            variable_name,
            entry_point_name
        )),
    }
}
