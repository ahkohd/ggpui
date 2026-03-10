#[cfg(not(feature = "video-ffmpeg"))]
fn main() {
    eprintln!(
        "This example requires the 'video-ffmpeg' feature.\n\nRun:\n  cargo run -p gpui --features video-ffmpeg --example custom_draw_api_video"
    );
}

#[cfg(feature = "video-ffmpeg")]
mod enabled {
    use std::path::Path;
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError, TrySendError};
    use std::thread;
    use std::time::{Duration, Instant};

    use anyhow::{anyhow, bail};
    use ffmpeg::Codec;
    use ffmpeg::ffi;
    use ffmpeg_next as ffmpeg;
    use gpui::colors::Colors;
    use gpui::{
        App, AppContext, Bounds, Context, CustomAddressMode, CustomBindingDesc, CustomBindingKind,
        CustomBindingName, CustomBindingValue, CustomBufferDesc, CustomBufferId,
        CustomBufferSource, CustomDrawParams, CustomFilterMode, CustomPipelineDesc,
        CustomPipelineId, CustomPipelineState, CustomPrimitiveTopology, CustomSamplerDesc,
        CustomSamplerId, CustomTextureDesc, CustomTextureDimension, CustomTextureFormat,
        CustomTextureId, CustomTextureUpdate, CustomTextureUsage, CustomVertexAttribute,
        CustomVertexAttributeName, CustomVertexBuffer, CustomVertexFetch, CustomVertexFormat,
        CustomVertexLayout, Hsla, InteractiveElement, Render, Styled, Window, WindowBounds,
        WindowOptions, canvas, div, prelude::*, px, size,
    };
    use gpui_platform::application;

    const SHADER_SOURCE: &str = r#"
struct VertexInput {
  a0: vec2<f32>,
  a1: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

var b0: texture_2d<f32>;
var b1: texture_2d<f32>;
var b2: sampler;

const YCBCR_TO_RGB: mat3x3<f32> = mat3x3<f32>(
  vec3<f32>(1.0000, 1.0000, 1.0000),
  vec3<f32>(0.0000, -0.3441, 1.7720),
  vec3<f32>(1.4020, -0.7141, 0.0000),
);

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4<f32>(input.a0, 0.0, 1.0);
  out.uv = input.a1;
  return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let y = textureSample(b0, b2, input.uv).r;
  let chroma = textureSample(b1, b2, input.uv).rg - vec2<f32>(0.5, 0.5);
  let rgb = clamp(YCBCR_TO_RGB * vec3<f32>(y, chroma), vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(rgb, 1.0);
}
"#;

    const DEFAULT_TARGET_FPS: f32 = 60.0;
    const VIDEO_VIEW_WIDTH: f32 = 640.0;
    const VIDEO_VIEW_HEIGHT: f32 = 420.0;
    const SCRUB_PREVIEW_SECONDS_BUDGET: f32 = 2.0;

    #[derive(Clone)]
    struct VideoExampleConfig {
        video_path: String,
        target_fps: f32,
        loop_playback: bool,
        allow_hardware_acceleration: bool,
    }

    impl VideoExampleConfig {
        fn from_args() -> Self {
            let mut config = Self {
                video_path: default_video_path(),
                target_fps: DEFAULT_TARGET_FPS,
                loop_playback: true,
                allow_hardware_acceleration: true,
            };

            let mut arguments = std::env::args().skip(1);
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--video" => {
                        if let Some(path) = arguments.next() {
                            config.video_path = path;
                        }
                    }
                    "--fps" => {
                        if let Some(value) = arguments.next()
                            && let Ok(parsed_value) = value.parse::<f32>()
                            && parsed_value.is_finite()
                            && parsed_value > 0.0
                        {
                            config.target_fps = parsed_value;
                        }
                    }
                    "--no-loop" => {
                        config.loop_playback = false;
                    }
                    "--software" => {
                        config.allow_hardware_acceleration = false;
                    }
                    _ => {}
                }
            }

            config
        }

        fn frame_interval(&self) -> Duration {
            Duration::from_secs_f32(1.0 / self.target_fps.max(1.0))
        }
    }

    fn default_video_path() -> String {
        format!(
            "{}/examples/assets/bird_60fps.mp4",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[derive(Clone)]
    struct DecodedVideoFrame {
        width: u32,
        height: u32,
        luma_width: u32,
        luma_height: u32,
        luma_bytes_per_row: u32,
        luma_data: Arc<[u8]>,
        chroma_width: u32,
        chroma_height: u32,
        chroma_bytes_per_row: u32,
        chroma_data: Arc<[u8]>,
        presentation_seconds: Option<f64>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct VideoFrameLayout {
        width: u32,
        height: u32,
        luma_width: u32,
        luma_height: u32,
        chroma_width: u32,
        chroma_height: u32,
    }

    impl VideoFrameLayout {
        fn from_frame(frame: &DecodedVideoFrame) -> Self {
            Self {
                width: frame.width,
                height: frame.height,
                luma_width: frame.luma_width,
                luma_height: frame.luma_height,
                chroma_width: frame.chroma_width,
                chroma_height: frame.chroma_height,
            }
        }
    }

    struct CachedNv12Scaler {
        source_format: ffmpeg::format::Pixel,
        source_width: u32,
        source_height: u32,
        context: ffmpeg::software::scaling::context::Context,
    }

    #[derive(Clone, Copy)]
    struct HardwareDecoderSetup {
        pixel_format: ffmpeg::format::Pixel,
        backend_name: &'static str,
    }

    struct VideoDecoder {
        input_context: ffmpeg::format::context::Input,
        video_stream_index: usize,
        stream_time_base: ffmpeg::Rational,
        stream_start_time: Option<i64>,
        stream_duration_seconds: Option<f64>,
        decoder: ffmpeg::decoder::Video,
        decoded_frame: ffmpeg::util::frame::video::Video,
        software_frame: ffmpeg::util::frame::video::Video,
        nv12_frame: ffmpeg::util::frame::video::Video,
        nv12_scaler: Option<CachedNv12Scaler>,
        hardware_pixel_format: Option<ffmpeg::format::Pixel>,
        hardware_backend_name: Option<&'static str>,
        drained: bool,
    }

    impl VideoDecoder {
        fn open(video_path: &str, allow_hardware_acceleration: bool) -> anyhow::Result<Self> {
            ffmpeg::init().map_err(|error| anyhow!("ffmpeg initialization failed: {error}"))?;

            let input_context = ffmpeg::format::input(video_path)
                .map_err(|error| anyhow!("failed to open video '{video_path}': {error}"))?;
            let input_stream = input_context
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or_else(|| anyhow!("no video stream found in '{video_path}'"))?;
            let video_stream_index = input_stream.index();
            let stream_time_base = input_stream.time_base();
            let stream_start_time = {
                let start_time = input_stream.start_time();
                if start_time == ffi::AV_NOPTS_VALUE {
                    None
                } else {
                    Some(start_time)
                }
            };
            let stream_duration_seconds =
                timestamp_to_seconds(input_stream.duration(), stream_time_base);

            let codec_id = input_stream.parameters().id();
            let codec = ffmpeg::codec::decoder::find(codec_id)
                .ok_or_else(|| anyhow!("failed to find decoder for codec id {codec_id:?}"))?;

            let mut hardware_setup = None;
            let decoder_result = {
                let mut decoder_context =
                    ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
                        .map_err(|error| {
                            anyhow!("failed to create video decoder context: {error}")
                        })?;
                if allow_hardware_acceleration {
                    hardware_setup = try_enable_hardware_decoder(&mut decoder_context, codec);
                }
                open_decoder(decoder_context, codec)
            };

            let decoder = match decoder_result {
                Ok(decoder) => decoder,
                Err(open_error) => {
                    if hardware_setup.is_some() {
                        log::warn!(
                            "hardware decoder setup failed for '{}': {open_error}. falling back to software decode",
                            video_path
                        );
                        hardware_setup = None;
                        let decoder_context = ffmpeg::codec::context::Context::from_parameters(
                            input_stream.parameters(),
                        )
                        .map_err(|error| {
                            anyhow!("failed to create software decoder context: {error}")
                        })?;
                        open_decoder(decoder_context, codec)?
                    } else {
                        return Err(open_error);
                    }
                }
            };

            let width = decoder.width();
            let height = decoder.height();
            if width == 0 || height == 0 {
                bail!("video decoder returned an invalid frame size ({width}x{height})");
            }

            Ok(Self {
                input_context,
                video_stream_index,
                stream_time_base,
                stream_start_time,
                stream_duration_seconds,
                decoder,
                decoded_frame: ffmpeg::util::frame::video::Video::empty(),
                software_frame: ffmpeg::util::frame::video::Video::empty(),
                nv12_frame: ffmpeg::util::frame::video::Video::empty(),
                nv12_scaler: None,
                hardware_pixel_format: hardware_setup.map(|setup| setup.pixel_format),
                hardware_backend_name: hardware_setup.map(|setup| setup.backend_name),
                drained: false,
            })
        }

        fn using_hardware_acceleration(&self) -> bool {
            self.hardware_backend_name.is_some()
        }

        fn hardware_backend_name(&self) -> Option<&'static str> {
            self.hardware_backend_name
        }

        fn duration_seconds(&self) -> Option<f64> {
            self.stream_duration_seconds
        }

        fn seek_to_seconds(&mut self, target_seconds: f64) -> anyhow::Result<()> {
            let clamped_seconds = if let Some(duration_seconds) = self.stream_duration_seconds {
                target_seconds.clamp(0.0, duration_seconds.max(0.0))
            } else {
                target_seconds.max(0.0)
            };
            let target_timestamp = (clamped_seconds * 1_000_000.0).round() as i64;

            self.input_context
                .seek(target_timestamp, ..)
                .map_err(|error| anyhow!("video seek failed: {error}"))?;
            self.decoder.flush();
            self.drained = false;
            Ok(())
        }

        fn seek_preview_frame(
            &mut self,
            target_seconds: f64,
            max_decode_frames: usize,
        ) -> anyhow::Result<Option<DecodedVideoFrame>> {
            self.seek_to_seconds(target_seconds)?;

            let mut latest_frame = None;
            let mut decoded_frames = 0usize;
            while decoded_frames < max_decode_frames.max(1) {
                let Some(decoded_frame) = self.next_frame()? else {
                    break;
                };

                let reached_target = decoded_frame
                    .presentation_seconds
                    .is_some_and(|presentation_seconds| presentation_seconds >= target_seconds);
                latest_frame = Some(decoded_frame);
                decoded_frames = decoded_frames.saturating_add(1);
                if reached_target {
                    break;
                }
            }

            Ok(latest_frame)
        }

        fn current_frame_presentation_seconds(&self) -> Option<f64> {
            let raw_timestamp = self
                .decoded_frame
                .timestamp()
                .or_else(|| self.decoded_frame.pts());
            let normalized_timestamp =
                normalize_stream_timestamp(raw_timestamp, self.stream_start_time)?;
            timestamp_to_seconds(normalized_timestamp, self.stream_time_base)
        }

        fn next_frame(&mut self) -> anyhow::Result<Option<DecodedVideoFrame>> {
            loop {
                if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                    return self.decode_frame_to_nv12();
                }

                if self.drained {
                    return Ok(None);
                }

                let mut submitted_packet = false;
                for (stream, packet) in self.input_context.packets() {
                    if stream.index() != self.video_stream_index {
                        continue;
                    }

                    self.decoder.send_packet(&packet).map_err(|error| {
                        anyhow!("video decode packet submission failed: {error}")
                    })?;
                    submitted_packet = true;
                    break;
                }

                if !submitted_packet {
                    self.decoder
                        .send_eof()
                        .map_err(|error| anyhow!("video decode drain failed: {error}"))?;
                    self.drained = true;
                }
            }
        }

        fn decode_frame_to_nv12(&mut self) -> anyhow::Result<Option<DecodedVideoFrame>> {
            let presentation_seconds = self.current_frame_presentation_seconds();
            if let Some(hardware_pixel_format) = self.hardware_pixel_format
                && self.decoded_frame.format() == hardware_pixel_format
            {
                unsafe {
                    ffi::av_frame_unref(self.software_frame.as_mut_ptr());
                }

                let transfer_status = unsafe {
                    ffi::av_hwframe_transfer_data(
                        self.software_frame.as_mut_ptr(),
                        self.decoded_frame.as_ptr(),
                        0,
                    )
                };
                if transfer_status < 0 {
                    bail!(
                        "hardware frame download failed with status {}",
                        transfer_status
                    );
                }

                return self.prepare_upload_frame_from_software(presentation_seconds);
            }

            self.prepare_upload_frame_from_decoded(presentation_seconds)
        }

        fn prepare_upload_frame_from_decoded(
            &mut self,
            presentation_seconds: Option<f64>,
        ) -> anyhow::Result<Option<DecodedVideoFrame>> {
            if self.decoded_frame.format() == ffmpeg::format::Pixel::NV12 {
                return Ok(Some(nv12_frame_to_upload(
                    &self.decoded_frame,
                    presentation_seconds,
                )?));
            }

            self.ensure_nv12_scaler(
                self.decoded_frame.format(),
                self.decoded_frame.width(),
                self.decoded_frame.height(),
            )?;

            let Some(nv12_scaler) = self.nv12_scaler.as_mut() else {
                bail!("NV12 scaler was not initialized")
            };
            nv12_scaler
                .context
                .run(&self.decoded_frame, &mut self.nv12_frame)
                .map_err(|error| anyhow!("NV12 conversion failed: {error}"))?;
            Ok(Some(nv12_frame_to_upload(
                &self.nv12_frame,
                presentation_seconds,
            )?))
        }

        fn prepare_upload_frame_from_software(
            &mut self,
            presentation_seconds: Option<f64>,
        ) -> anyhow::Result<Option<DecodedVideoFrame>> {
            if self.software_frame.format() == ffmpeg::format::Pixel::NV12 {
                return Ok(Some(nv12_frame_to_upload(
                    &self.software_frame,
                    presentation_seconds,
                )?));
            }

            self.ensure_nv12_scaler(
                self.software_frame.format(),
                self.software_frame.width(),
                self.software_frame.height(),
            )?;

            let Some(nv12_scaler) = self.nv12_scaler.as_mut() else {
                bail!("NV12 scaler was not initialized")
            };
            nv12_scaler
                .context
                .run(&self.software_frame, &mut self.nv12_frame)
                .map_err(|error| anyhow!("NV12 conversion failed: {error}"))?;
            Ok(Some(nv12_frame_to_upload(
                &self.nv12_frame,
                presentation_seconds,
            )?))
        }

        fn ensure_nv12_scaler(
            &mut self,
            source_format: ffmpeg::format::Pixel,
            source_width: u32,
            source_height: u32,
        ) -> anyhow::Result<()> {
            let needs_new_scaler = self.nv12_scaler.as_ref().is_none_or(|cached_scaler| {
                cached_scaler.source_format != source_format
                    || cached_scaler.source_width != source_width
                    || cached_scaler.source_height != source_height
            });

            if needs_new_scaler {
                let context = ffmpeg::software::scaling::context::Context::get(
                    source_format,
                    source_width,
                    source_height,
                    ffmpeg::format::Pixel::NV12,
                    source_width,
                    source_height,
                    ffmpeg::software::scaling::flag::Flags::BILINEAR,
                )
                .map_err(|error| anyhow!("failed to create NV12 scaler: {error}"))?;
                self.nv12_scaler = Some(CachedNv12Scaler {
                    source_format,
                    source_width,
                    source_height,
                    context,
                });
            }

            Ok(())
        }
    }

    fn open_decoder(
        decoder_context: ffmpeg::codec::context::Context,
        codec: Codec,
    ) -> anyhow::Result<ffmpeg::decoder::Video> {
        decoder_context
            .decoder()
            .open_as(codec)
            .and_then(|opened_decoder| opened_decoder.video())
            .map_err(|error| anyhow!("failed to open video decoder: {error}"))
    }

    fn try_enable_hardware_decoder(
        decoder_context: &mut ffmpeg::codec::context::Context,
        codec: Codec,
    ) -> Option<HardwareDecoderSetup> {
        for (device_type, backend_name) in preferred_hardware_device_types() {
            let Some(hardware_pixel_format) =
                (unsafe { find_hardware_pixel_format(codec, *device_type) })
            else {
                continue;
            };

            let mut hardware_device_context: *mut ffi::AVBufferRef = ptr::null_mut();
            let create_result = unsafe {
                ffi::av_hwdevice_ctx_create(
                    &mut hardware_device_context,
                    *device_type,
                    ptr::null(),
                    ptr::null_mut(),
                    0,
                )
            };
            if create_result < 0 || hardware_device_context.is_null() {
                continue;
            }

            let decoder_context_ref = unsafe { decoder_context.as_mut_ptr() };
            unsafe {
                (*decoder_context_ref).hw_device_ctx = ffi::av_buffer_ref(hardware_device_context);
                ffi::av_buffer_unref(&mut hardware_device_context);
            }

            let has_context = unsafe { !(*decoder_context_ref).hw_device_ctx.is_null() };
            if !has_context {
                continue;
            }

            return Some(HardwareDecoderSetup {
                pixel_format: ffmpeg::format::Pixel::from(hardware_pixel_format),
                backend_name,
            });
        }

        None
    }

    unsafe fn find_hardware_pixel_format(
        codec: Codec,
        device_type: ffi::AVHWDeviceType,
    ) -> Option<ffi::AVPixelFormat> {
        let mut config_index = 0;
        loop {
            let hardware_config =
                unsafe { ffi::avcodec_get_hw_config(codec.as_ptr(), config_index) };
            if hardware_config.is_null() {
                return None;
            }

            let supports_device_context = unsafe {
                ((*hardware_config).methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32)
                    != 0
            };
            let config_device_type = unsafe { (*hardware_config).device_type };
            if supports_device_context && config_device_type == device_type {
                return Some(unsafe { (*hardware_config).pix_fmt });
            }

            config_index += 1;
        }
    }

    fn normalize_stream_timestamp(
        raw_timestamp: Option<i64>,
        stream_start_time: Option<i64>,
    ) -> Option<i64> {
        let timestamp = raw_timestamp?;
        if timestamp == ffi::AV_NOPTS_VALUE {
            return None;
        }

        if let Some(start_time) = stream_start_time {
            if start_time != ffi::AV_NOPTS_VALUE {
                return Some(timestamp.saturating_sub(start_time));
            }
        }

        Some(timestamp)
    }

    fn timestamp_to_seconds(timestamp: i64, time_base: ffmpeg::Rational) -> Option<f64> {
        if timestamp == ffi::AV_NOPTS_VALUE {
            return None;
        }

        let time_base_denominator = time_base.denominator();
        if time_base_denominator == 0 {
            return None;
        }

        let seconds =
            (timestamp as f64) * (time_base.numerator() as f64) / (time_base_denominator as f64);
        if seconds.is_finite() {
            Some(seconds.max(0.0))
        } else {
            None
        }
    }

    #[cfg(target_os = "macos")]
    fn preferred_hardware_device_types() -> &'static [(ffi::AVHWDeviceType, &'static str)] {
        &[(
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            "videotoolbox",
        )]
    }

    #[cfg(target_os = "windows")]
    fn preferred_hardware_device_types() -> &'static [(ffi::AVHWDeviceType, &'static str)] {
        &[(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA, "d3d11va")]
    }

    #[cfg(target_os = "linux")]
    fn preferred_hardware_device_types() -> &'static [(ffi::AVHWDeviceType, &'static str)] {
        &[(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI, "vaapi")]
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    fn preferred_hardware_device_types() -> &'static [(ffi::AVHWDeviceType, &'static str)] {
        &[]
    }

    fn nv12_frame_to_upload(
        frame: &ffmpeg::util::frame::video::Video,
        presentation_seconds: Option<f64>,
    ) -> anyhow::Result<DecodedVideoFrame> {
        if frame.format() != ffmpeg::format::Pixel::NV12 {
            bail!("expected NV12 frame, got {:?}", frame.format());
        }
        if frame.planes() < 2 {
            bail!("NV12 frame is missing chroma plane data");
        }

        let width = frame.width();
        let height = frame.height();
        if width == 0 || height == 0 {
            bail!("decoded frame has invalid dimensions ({width}x{height})");
        }

        let luma_width = frame.plane_width(0);
        let luma_height = frame.plane_height(0);
        let luma_stride = frame.stride(0);
        let luma_row_bytes =
            usize::try_from(luma_width).map_err(|_| anyhow!("invalid luma width {luma_width}"))?;

        let chroma_width = frame.plane_width(1);
        let chroma_height = frame.plane_height(1);
        let chroma_stride = frame.stride(1);
        let chroma_row_bytes = usize::try_from(chroma_width)
            .map_err(|_| anyhow!("invalid chroma width {chroma_width}"))?
            .checked_mul(2)
            .ok_or_else(|| anyhow!("chroma row width overflow"))?;

        if luma_stride < luma_row_bytes {
            bail!(
                "luma stride {} is smaller than expected row size {}",
                luma_stride,
                luma_row_bytes
            );
        }
        if chroma_stride < chroma_row_bytes {
            bail!(
                "chroma stride {} is smaller than expected row size {}",
                chroma_stride,
                chroma_row_bytes
            );
        }

        let luma_height_usize =
            usize::try_from(luma_height).map_err(|_| anyhow!("invalid luma height"))?;
        let chroma_height_usize =
            usize::try_from(chroma_height).map_err(|_| anyhow!("invalid chroma height"))?;

        let mut luma_data = vec![0u8; luma_row_bytes.saturating_mul(luma_height_usize)];
        let source_luma_data = frame.data(0);
        for row_index in 0..luma_height_usize {
            let source_start = row_index
                .checked_mul(luma_stride)
                .ok_or_else(|| anyhow!("luma source row overflow"))?;
            let source_end = source_start
                .checked_add(luma_row_bytes)
                .ok_or_else(|| anyhow!("luma source slice overflow"))?;
            if source_end > source_luma_data.len() {
                bail!("luma source row {} exceeds source bounds", row_index);
            }

            let destination_start = row_index
                .checked_mul(luma_row_bytes)
                .ok_or_else(|| anyhow!("luma destination row overflow"))?;
            let destination_end = destination_start
                .checked_add(luma_row_bytes)
                .ok_or_else(|| anyhow!("luma destination slice overflow"))?;
            luma_data[destination_start..destination_end]
                .copy_from_slice(&source_luma_data[source_start..source_end]);
        }

        let mut chroma_data = vec![0u8; chroma_row_bytes.saturating_mul(chroma_height_usize)];
        let source_chroma_data = frame.data(1);
        for row_index in 0..chroma_height_usize {
            let source_start = row_index
                .checked_mul(chroma_stride)
                .ok_or_else(|| anyhow!("chroma source row overflow"))?;
            let source_end = source_start
                .checked_add(chroma_row_bytes)
                .ok_or_else(|| anyhow!("chroma source slice overflow"))?;
            if source_end > source_chroma_data.len() {
                bail!("chroma source row {} exceeds source bounds", row_index);
            }

            let destination_start = row_index
                .checked_mul(chroma_row_bytes)
                .ok_or_else(|| anyhow!("chroma destination row overflow"))?;
            let destination_end = destination_start
                .checked_add(chroma_row_bytes)
                .ok_or_else(|| anyhow!("chroma destination slice overflow"))?;
            chroma_data[destination_start..destination_end]
                .copy_from_slice(&source_chroma_data[source_start..source_end]);
        }

        Ok(DecodedVideoFrame {
            width,
            height,
            luma_width,
            luma_height,
            luma_bytes_per_row: u32::try_from(luma_row_bytes)
                .map_err(|_| anyhow!("luma row size does not fit into u32"))?,
            luma_data: Arc::from(luma_data),
            chroma_width,
            chroma_height,
            chroma_bytes_per_row: u32::try_from(chroma_row_bytes)
                .map_err(|_| anyhow!("chroma row size does not fit into u32"))?,
            chroma_data: Arc::from(chroma_data),
            presentation_seconds,
        })
    }

    enum DecodeMessage {
        Frame(DecodedVideoFrame),
        Ended,
        Error(String),
    }

    enum DecodeCommand {
        SetPaused(bool),
        SetLoopPlayback(bool),
        SeekSeconds(f64),
        Restart,
    }

    struct VideoDecodeWorker {
        receiver: Receiver<DecodeMessage>,
        command_sender: Sender<DecodeCommand>,
        stop_signal: Arc<AtomicBool>,
        join_handle: Option<thread::JoinHandle<()>>,
    }

    impl VideoDecodeWorker {
        fn spawn(
            video_path: String,
            loop_playback: bool,
            allow_hardware_acceleration: bool,
            frame_interval: Duration,
        ) -> anyhow::Result<Self> {
            let (sender, receiver) = mpsc::sync_channel(3);
            let (command_sender, command_receiver) = mpsc::channel();
            let stop_signal = Arc::new(AtomicBool::new(false));
            let thread_stop_signal = stop_signal.clone();

            let join_handle = thread::Builder::new()
                .name("custom-draw-video-decode".to_string())
                .spawn(move || {
                    let mut video_decoder =
                        match VideoDecoder::open(&video_path, allow_hardware_acceleration) {
                            Ok(video_decoder) => video_decoder,
                            Err(error) => {
                                let _ = send_decode_terminal_message(
                                    &sender,
                                    DecodeMessage::Error(error.to_string()),
                                );
                                return;
                            }
                        };

                    let mut paused = false;
                    let mut playback_ended = false;
                    let mut loop_playback = loop_playback;
                    let frame_interval_seconds = frame_interval.as_secs_f32().max(1.0 / 240.0);
                    let scrub_preview_max_decode_frames =
                        ((SCRUB_PREVIEW_SECONDS_BUDGET / frame_interval_seconds).ceil() as usize)
                            .clamp(24, 240);
                    let mut ended_notified = false;
                    let mut last_frame_time = Instant::now() - frame_interval;

                    while !thread_stop_signal.load(Ordering::Relaxed) {
                        let mut latest_seek_seconds = None;
                        loop {
                            match command_receiver.try_recv() {
                                Ok(DecodeCommand::SetPaused(next_paused)) => {
                                    paused = next_paused;
                                    if !paused {
                                        last_frame_time = Instant::now() - frame_interval;
                                    }
                                }
                                Ok(DecodeCommand::SetLoopPlayback(next_loop_playback)) => {
                                    loop_playback = next_loop_playback;
                                    if loop_playback && playback_ended {
                                        match VideoDecoder::open(
                                            &video_path,
                                            allow_hardware_acceleration,
                                        ) {
                                            Ok(next_video_decoder) => {
                                                video_decoder = next_video_decoder;
                                                playback_ended = false;
                                                ended_notified = false;
                                                last_frame_time = Instant::now() - frame_interval;
                                            }
                                            Err(error) => {
                                                let _ = send_decode_terminal_message(
                                                    &sender,
                                                    DecodeMessage::Error(error.to_string()),
                                                );
                                                return;
                                            }
                                        }
                                    }
                                }
                                Ok(DecodeCommand::SeekSeconds(target_seconds)) => {
                                    latest_seek_seconds = Some(target_seconds);
                                }
                                Ok(DecodeCommand::Restart) => {
                                    match VideoDecoder::open(
                                        &video_path,
                                        allow_hardware_acceleration,
                                    ) {
                                        Ok(next_video_decoder) => {
                                            video_decoder = next_video_decoder;
                                            playback_ended = false;
                                            ended_notified = false;
                                            paused = false;
                                            last_frame_time = Instant::now() - frame_interval;
                                        }
                                        Err(error) => {
                                            let _ = send_decode_terminal_message(
                                                &sender,
                                                DecodeMessage::Error(error.to_string()),
                                            );
                                            return;
                                        }
                                    }
                                }
                                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {
                                    break;
                                }
                            }
                        }

                        if let Some(target_seconds) = latest_seek_seconds {
                            match video_decoder
                                .seek_preview_frame(target_seconds, scrub_preview_max_decode_frames)
                            {
                                Ok(Some(decoded_frame)) => {
                                    playback_ended = false;
                                    ended_notified = false;
                                    last_frame_time = Instant::now() - frame_interval;
                                    if send_decode_frame(&sender, decoded_frame) {
                                        return;
                                    }
                                }
                                Ok(None) => {
                                    playback_ended = true;
                                    ended_notified = false;
                                }
                                Err(error) => {
                                    let _ = send_decode_terminal_message(
                                        &sender,
                                        DecodeMessage::Error(error.to_string()),
                                    );
                                    return;
                                }
                            }
                            continue;
                        }

                        if playback_ended {
                            if loop_playback {
                                match VideoDecoder::open(&video_path, allow_hardware_acceleration) {
                                    Ok(next_video_decoder) => {
                                        video_decoder = next_video_decoder;
                                        playback_ended = false;
                                        ended_notified = false;
                                        last_frame_time = Instant::now() - frame_interval;
                                    }
                                    Err(error) => {
                                        let _ = send_decode_terminal_message(
                                            &sender,
                                            DecodeMessage::Error(error.to_string()),
                                        );
                                        return;
                                    }
                                }
                                continue;
                            }

                            if !ended_notified {
                                if send_decode_terminal_message(&sender, DecodeMessage::Ended) {
                                    return;
                                }
                                ended_notified = true;
                            }
                            thread::sleep(Duration::from_millis(16));
                            continue;
                        }

                        if paused {
                            thread::sleep(Duration::from_millis(8));
                            continue;
                        }

                        match video_decoder.next_frame() {
                            Ok(Some(decoded_frame)) => {
                                let elapsed = last_frame_time.elapsed();
                                if elapsed < frame_interval {
                                    thread::sleep(frame_interval - elapsed);
                                }
                                last_frame_time = Instant::now();
                                if send_decode_frame(&sender, decoded_frame) {
                                    return;
                                }
                            }
                            Ok(None) => {
                                if loop_playback {
                                    match VideoDecoder::open(
                                        &video_path,
                                        allow_hardware_acceleration,
                                    ) {
                                        Ok(next_video_decoder) => {
                                            video_decoder = next_video_decoder;
                                            playback_ended = false;
                                            ended_notified = false;
                                            last_frame_time = Instant::now() - frame_interval;
                                        }
                                        Err(error) => {
                                            let _ = send_decode_terminal_message(
                                                &sender,
                                                DecodeMessage::Error(error.to_string()),
                                            );
                                            return;
                                        }
                                    }
                                } else {
                                    playback_ended = true;
                                }
                            }
                            Err(error) => {
                                let _ = send_decode_terminal_message(
                                    &sender,
                                    DecodeMessage::Error(error.to_string()),
                                );
                                return;
                            }
                        }
                    }
                })
                .map_err(|error| anyhow!("failed to spawn decode thread: {error}"))?;

            Ok(Self {
                receiver,
                command_sender,
                stop_signal,
                join_handle: Some(join_handle),
            })
        }

        fn set_paused(&self, paused: bool) {
            if let Err(error) = self.command_sender.send(DecodeCommand::SetPaused(paused)) {
                log::error!("failed to send video decode pause command: {error}");
            }
        }

        fn set_loop_playback(&self, loop_playback: bool) {
            if let Err(error) = self
                .command_sender
                .send(DecodeCommand::SetLoopPlayback(loop_playback))
            {
                log::error!("failed to send video decode loop command: {error}");
            }
        }

        fn seek_seconds(&self, target_seconds: f64) {
            if let Err(error) = self
                .command_sender
                .send(DecodeCommand::SeekSeconds(target_seconds))
            {
                log::error!("failed to send video decode seek command: {error}");
            }
        }

        fn restart(&self) {
            if let Err(error) = self.command_sender.send(DecodeCommand::Restart) {
                log::error!("failed to send video decode restart command: {error}");
            }
        }

        fn drain_latest_message(&mut self) -> Option<DecodeMessage> {
            let mut latest_message = None;
            loop {
                match self.receiver.try_recv() {
                    Ok(message) => {
                        latest_message = Some(message);
                    }
                    Err(TryRecvError::Empty) => {
                        return latest_message;
                    }
                    Err(TryRecvError::Disconnected) => {
                        if latest_message.is_none() {
                            latest_message = Some(DecodeMessage::Error(
                                "video decode thread disconnected unexpectedly".to_string(),
                            ));
                        }
                        return latest_message;
                    }
                }
            }
        }

        fn stop(&mut self) {
            self.stop_signal.store(true, Ordering::Relaxed);
            if let Some(join_handle) = self.join_handle.take()
                && let Err(join_error) = join_handle.join()
            {
                log::error!("video decode thread join failed: {join_error:?}");
            }
        }
    }

    impl Drop for VideoDecodeWorker {
        fn drop(&mut self) {
            self.stop();
        }
    }

    fn send_decode_frame(sender: &SyncSender<DecodeMessage>, frame: DecodedVideoFrame) -> bool {
        match sender.try_send(DecodeMessage::Frame(frame)) {
            Ok(()) => false,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => true,
        }
    }

    fn send_decode_terminal_message(
        sender: &SyncSender<DecodeMessage>,
        message: DecodeMessage,
    ) -> bool {
        sender.send(message).is_err()
    }

    struct VideoResources {
        pipeline: CustomPipelineId,
        vertex_buffer: CustomBufferId,
        luma_texture: CustomTextureId,
        chroma_texture: CustomTextureId,
        sampler: CustomSamplerId,
        frame_layout: VideoFrameLayout,
        duration_seconds: Option<f64>,
        decode_worker: VideoDecodeWorker,
        using_hardware_acceleration: bool,
        hardware_backend_name: Option<String>,
    }

    struct VideoCustomDrawExample {
        config: VideoExampleConfig,
        frame_interval: Duration,
        loop_playback: bool,
        paused: bool,
        playback_ended: bool,
        scrubbing: bool,
        resume_playback_after_scrub: bool,
        current_position_seconds: f64,
        resources: Option<VideoResources>,
        error: Option<String>,
    }

    impl VideoCustomDrawExample {
        fn new(config: VideoExampleConfig, _cx: &mut Context<Self>) -> Self {
            let loop_playback = config.loop_playback;
            Self {
                frame_interval: config.frame_interval(),
                config,
                loop_playback,
                paused: false,
                playback_ended: false,
                scrubbing: false,
                resume_playback_after_scrub: false,
                current_position_seconds: 0.0,
                resources: None,
                error: None,
            }
        }

        fn ensure_resources(&mut self, window: &mut Window) {
            if self.resources.is_some() || self.error.is_some() {
                return;
            }

            match self.build_resources(window) {
                Ok(resources) => {
                    self.resources = Some(resources);
                }
                Err(error) => {
                    self.error = Some(error.to_string());
                }
            }
        }

        fn build_resources(&mut self, window: &mut Window) -> anyhow::Result<VideoResources> {
            if !Path::new(&self.config.video_path).exists() {
                bail!(
                    "video file does not exist: {} (pass --video /path/to/video.mp4 to override)",
                    self.config.video_path
                );
            }

            let mut video_decoder = VideoDecoder::open(
                &self.config.video_path,
                self.config.allow_hardware_acceleration,
            )?;
            let duration_seconds = video_decoder.duration_seconds();
            let first_frame = video_decoder.next_frame()?.ok_or_else(|| {
                anyhow!("video has no decodable frames: {}", self.config.video_path)
            })?;
            if let Some(first_frame_presentation_seconds) = first_frame.presentation_seconds {
                self.current_position_seconds = first_frame_presentation_seconds;
            }
            let frame_layout = VideoFrameLayout::from_frame(&first_frame);

            let pipeline = window.create_custom_pipeline(CustomPipelineDesc {
                name: "custom_draw_video_pipeline".to_string(),
                shader_source: SHADER_SOURCE.to_string(),
                vertex_entry: "vs_main".to_string(),
                fragment_entry: "fs_main".to_string(),
                vertex_fetches: vec![CustomVertexFetch {
                    layout: CustomVertexLayout {
                        stride: 16,
                        attributes: vec![
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A0,
                                offset: 0,
                                format: CustomVertexFormat::F32Vec2,
                                location: None,
                            },
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A1,
                                offset: 8,
                                format: CustomVertexFormat::F32Vec2,
                                location: None,
                            },
                        ],
                    },
                    instanced: false,
                }],
                primitive: CustomPrimitiveTopology::TriangleList,
                color_targets: Vec::new(),
                state: CustomPipelineState::default(),
                push_constants: None,
                bindings: vec![
                    CustomBindingDesc {
                        name: CustomBindingName::B0,
                        kind: CustomBindingKind::Texture,
                        slot: None,
                    },
                    CustomBindingDesc {
                        name: CustomBindingName::B1,
                        kind: CustomBindingKind::Texture,
                        slot: None,
                    },
                    CustomBindingDesc {
                        name: CustomBindingName::B2,
                        kind: CustomBindingKind::Sampler,
                        slot: None,
                    },
                ],
            })?;

            let vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
                name: "custom_draw_video_quad_vertices".to_string(),
                data: quad_vertex_data(),
            })?;

            let (luma_texture, chroma_texture) = create_video_textures(window, &first_frame)?;

            let sampler = window.create_custom_sampler(CustomSamplerDesc {
                name: "custom_draw_video_sampler".to_string(),
                min_filter: CustomFilterMode::Linear,
                mag_filter: CustomFilterMode::Linear,
                mipmap_filter: CustomFilterMode::Nearest,
                address_modes: [CustomAddressMode::ClampToEdge; 3],
            })?;

            let using_hardware_acceleration = video_decoder.using_hardware_acceleration();
            let hardware_backend_name = video_decoder
                .hardware_backend_name()
                .map(ToString::to_string);

            let decode_worker = VideoDecodeWorker::spawn(
                self.config.video_path.clone(),
                self.loop_playback,
                self.config.allow_hardware_acceleration,
                self.frame_interval,
            )?;
            decode_worker.set_paused(self.paused);

            Ok(VideoResources {
                pipeline,
                vertex_buffer,
                luma_texture,
                chroma_texture,
                sampler,
                frame_layout,
                duration_seconds,
                decode_worker,
                using_hardware_acceleration,
                hardware_backend_name,
            })
        }

        fn advance_video_frame(&mut self, window: &mut Window) -> anyhow::Result<()> {
            let Some(resources) = self.resources.as_mut() else {
                return Ok(());
            };

            let Some(latest_message) = resources.decode_worker.drain_latest_message() else {
                return Ok(());
            };

            match latest_message {
                DecodeMessage::Frame(decoded_frame) => {
                    self.playback_ended = false;
                    if !self.scrubbing
                        && let Some(presentation_seconds) = decoded_frame.presentation_seconds
                    {
                        self.current_position_seconds = presentation_seconds;
                    }
                    update_video_textures(window, resources, decoded_frame)?;
                }
                DecodeMessage::Ended => {
                    self.playback_ended = true;
                    self.paused = true;
                    if let Some(duration_seconds) = resources.duration_seconds {
                        self.current_position_seconds = duration_seconds;
                    }
                }
                DecodeMessage::Error(error) => {
                    self.error = Some(error);
                }
            }

            Ok(())
        }

        fn toggle_playback(&mut self) {
            if self.playback_ended {
                self.restart_playback();
                return;
            }

            self.paused = !self.paused;
            self.scrubbing = false;
            self.resume_playback_after_scrub = false;
            if let Some(resources) = self.resources.as_ref() {
                resources.decode_worker.set_paused(self.paused);
            }
        }

        fn toggle_loop_playback(&mut self) {
            self.loop_playback = !self.loop_playback;
            self.resume_playback_after_scrub = false;
            if self.loop_playback && self.playback_ended {
                self.playback_ended = false;
                self.paused = false;
            }
            if let Some(resources) = self.resources.as_ref() {
                resources
                    .decode_worker
                    .set_loop_playback(self.loop_playback);
                if !self.paused {
                    resources.decode_worker.set_paused(false);
                }
            }
        }

        fn restart_playback(&mut self) {
            self.playback_ended = false;
            self.paused = false;
            self.scrubbing = false;
            self.resume_playback_after_scrub = false;
            self.current_position_seconds = 0.0;
            if let Some(resources) = self.resources.as_ref() {
                resources.decode_worker.restart();
                resources.decode_worker.set_paused(false);
            }
        }

        fn duration_seconds(&self) -> Option<f64> {
            self.resources
                .as_ref()
                .and_then(|resources| resources.duration_seconds)
        }

        fn seek_to_ratio(&mut self, ratio: f64) {
            let Some(duration_seconds) = self.duration_seconds() else {
                return;
            };

            let clamped_ratio = ratio.clamp(0.0, 1.0);
            let target_seconds = duration_seconds * clamped_ratio;
            self.current_position_seconds = target_seconds;
            self.playback_ended = false;
            if let Some(resources) = self.resources.as_ref() {
                resources.decode_worker.seek_seconds(target_seconds);
            }
        }

        fn seek_from_mouse_position(
            &mut self,
            mouse_position: gpui::Point<gpui::Pixels>,
            window: &mut Window,
        ) {
            let viewport_width = f32::from(window.viewport_size().width);
            if !viewport_width.is_finite() || viewport_width <= 0.0 {
                return;
            }

            let track_width = VIDEO_VIEW_WIDTH.max(1.0);
            let track_origin_x = ((viewport_width - track_width).max(0.0)) * 0.5;
            let clamped_x = (f32::from(mouse_position.x) - track_origin_x).clamp(0.0, track_width);
            let ratio = (clamped_x / track_width) as f64;
            self.seek_to_ratio(ratio);
        }

        fn begin_scrub(&mut self, mouse_position: gpui::Point<gpui::Pixels>, window: &mut Window) {
            if self.scrubbing {
                self.seek_from_mouse_position(mouse_position, window);
                return;
            }

            self.scrubbing = true;
            self.resume_playback_after_scrub = !self.paused && !self.playback_ended;
            if self.resume_playback_after_scrub {
                self.paused = true;
                if let Some(resources) = self.resources.as_ref() {
                    resources.decode_worker.set_paused(true);
                }
            }
            self.seek_from_mouse_position(mouse_position, window);
        }

        fn update_scrub(&mut self, mouse_position: gpui::Point<gpui::Pixels>, window: &mut Window) {
            if self.scrubbing {
                self.seek_from_mouse_position(mouse_position, window);
            }
        }

        fn end_scrub(&mut self) {
            if !self.scrubbing {
                return;
            }

            self.scrubbing = false;
            if self.resume_playback_after_scrub {
                self.paused = false;
                if let Some(resources) = self.resources.as_ref() {
                    resources.decode_worker.set_paused(false);
                }
            }
            self.resume_playback_after_scrub = false;
        }

        fn playback_status_text(&self) -> &'static str {
            if self.playback_ended {
                "Playback: ended"
            } else if self.paused {
                "Playback: paused"
            } else {
                "Playback: playing"
            }
        }

        fn decode_status_text(&self) -> String {
            if let Some(resources) = self.resources.as_ref() {
                if resources.using_hardware_acceleration {
                    return format!(
                        "Decoder: hardware ({})",
                        resources
                            .hardware_backend_name
                            .as_deref()
                            .unwrap_or("unknown")
                    );
                }
                return "Decoder: software".to_string();
            }

            if self.config.allow_hardware_acceleration {
                "Decoder: initializing (hardware auto)".to_string()
            } else {
                "Decoder: initializing (software forced)".to_string()
            }
        }
    }

    impl Render for VideoCustomDrawExample {
        fn render(
            &mut self,
            window: &mut Window,
            cx: &mut Context<Self>,
        ) -> impl gpui::IntoElement {
            let colors = Colors::for_appearance(window);
            self.ensure_resources(window);
            if self.error.is_none()
                && let Err(error) = self.advance_video_frame(window)
            {
                self.error = Some(error.to_string());
            }
            window.request_animation_frame();

            let duration_seconds = self.duration_seconds();
            let progress_ratio = duration_seconds
                .map(|duration| {
                    if duration <= 0.0 {
                        0.0
                    } else {
                        (self.current_position_seconds / duration).clamp(0.0, 1.0)
                    }
                })
                .unwrap_or(0.0);
            let timeline_text = if let Some(duration) = duration_seconds {
                format!(
                    "{} / {}",
                    format_clock_time(self.current_position_seconds),
                    format_clock_time(duration)
                )
            } else {
                format!(
                    "{} / --:--",
                    format_clock_time(self.current_position_seconds)
                )
            };
            let timeline_overlay_text = timeline_text.clone();

            let controls = div()
                .flex()
                .gap_2()
                .child(
                    div()
                        .id("video-toggle-play")
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .cursor_pointer()
                        .text_color(colors.selected_text)
                        .bg(colors.selected)
                        .hover(|style| style.bg(colors.selected))
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(|this, _: &gpui::MouseDownEvent, _, _| {
                                this.toggle_playback();
                            }),
                        )
                        .child(if self.playback_ended {
                            "Replay"
                        } else if self.paused {
                            "Play"
                        } else {
                            "Pause"
                        }),
                )
                .child(
                    div()
                        .id("video-restart")
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .cursor_pointer()
                        .text_color(colors.selected_text)
                        .bg(colors.selected)
                        .hover(|style| style.bg(colors.selected))
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(|this, _: &gpui::MouseDownEvent, _, _| {
                                this.restart_playback();
                            }),
                        )
                        .child("Restart"),
                )
                .child(
                    div()
                        .id("video-toggle-loop")
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .cursor_pointer()
                        .text_color(colors.selected_text)
                        .bg(colors.selected)
                        .hover(|style| style.bg(colors.selected))
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(|this, _: &gpui::MouseDownEvent, _, _| {
                                this.toggle_loop_playback();
                            }),
                        )
                        .child(if self.loop_playback {
                            "Loop: On"
                        } else {
                            "Loop: Off"
                        }),
                );

            let scrubber = div()
                .id("video-scrubber")
                .w(px(VIDEO_VIEW_WIDTH))
                .h(px(14.0))
                .rounded_md()
                .cursor_pointer()
                .bg(colors.container)
                .border_1()
                .border_color(colors.border)
                .on_mouse_down(
                    gpui::MouseButton::Left,
                    cx.listener(|this, event: &gpui::MouseDownEvent, window, _| {
                        this.begin_scrub(event.position, window);
                    }),
                )
                .child(
                    div()
                        .h_full()
                        .w(gpui::relative(progress_ratio as f32))
                        .bg(colors.selected)
                        .rounded_md(),
                );

            let header = div()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .text_xl()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(colors.text)
                        .child("Custom Draw API (Video)"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.disabled)
                        .child("Hardware decode + GPU YUV sampling with ffmpeg-next"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.disabled)
                        .child(self.decode_status_text()),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.disabled)
                        .child(self.playback_status_text()),
                )
                .child(div().text_sm().text_color(colors.text).child(timeline_text))
                .child(scrubber)
                .child(controls);

            let surface_color: Hsla = colors.container.into();
            let overlay_background: Hsla = colors.background.into();
            let content = if let Some(error) = &self.error {
                div()
                    .text_sm()
                    .text_color(gpui::red())
                    .child(format!("Custom draw/video unavailable: {error}"))
            } else if let Some(resources) = self.resources.as_ref() {
                let pipeline = resources.pipeline;
                let vertex_buffer = resources.vertex_buffer;
                let luma_texture = resources.luma_texture;
                let chroma_texture = resources.chroma_texture;
                let sampler = resources.sampler;

                let prepaint = move |bounds: Bounds<_>, window: &mut Window, _cx: &mut App| {
                    let vertices = quad_vertex_data_for_bounds(bounds, window.viewport_size());
                    if let Err(error) = window.update_custom_buffer(vertex_buffer, vertices) {
                        log::error!("video quad vertex update failed: {error}");
                    }

                    CustomDrawParams {
                        bounds,
                        pipeline,
                        vertex_buffers: vec![CustomVertexBuffer {
                            source: CustomBufferSource::Buffer(vertex_buffer),
                        }],
                        vertex_count: 6,
                        index_buffer: None,
                        index_count: 0,
                        target: None,
                        instance_count: 1,
                        push_constants: None,
                        bindings: vec![
                            CustomBindingValue::Texture(luma_texture),
                            CustomBindingValue::Texture(chroma_texture),
                            CustomBindingValue::Sampler(sampler),
                        ],
                    }
                };

                let paint = move |_bounds: Bounds<_>,
                                  params: CustomDrawParams,
                                  window: &mut Window,
                                  _cx: &mut App| {
                    if let Err(error) = window.paint_custom(params) {
                        log::error!("video custom draw paint failed: {error}");
                    }
                };

                div()
                    .w(px(VIDEO_VIEW_WIDTH))
                    .h(px(VIDEO_VIEW_HEIGHT))
                    .relative()
                    .rounded_md()
                    .border_1()
                    .border_color(colors.border)
                    .bg(surface_color.opacity(0.2))
                    .child(canvas(prepaint, paint).size_full())
                    .child(
                        div()
                            .absolute()
                            .top_2()
                            .right_2()
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .bg(overlay_background.opacity(0.85))
                            .text_sm()
                            .text_color(colors.text)
                            .child(timeline_overlay_text),
                    )
            } else {
                div()
                    .text_sm()
                    .text_color(colors.disabled)
                    .child("Initializing video playback resources...")
            };

            div()
                .size_full()
                .p_6()
                .bg(colors.background)
                .flex()
                .flex_col()
                .items_center()
                .on_mouse_move(
                    cx.listener(|this, event: &gpui::MouseMoveEvent, window, _| {
                        this.update_scrub(event.position, window);
                    }),
                )
                .on_mouse_up(
                    gpui::MouseButton::Left,
                    cx.listener(|this, _: &gpui::MouseUpEvent, _, _| {
                        this.end_scrub();
                    }),
                )
                .child(
                    div()
                        .w(px(VIDEO_VIEW_WIDTH))
                        .flex()
                        .flex_col()
                        .gap_4()
                        .child(header)
                        .child(content),
                )
        }
    }

    fn create_video_textures(
        window: &mut Window,
        frame: &DecodedVideoFrame,
    ) -> anyhow::Result<(CustomTextureId, CustomTextureId)> {
        let luma_texture = window.create_custom_texture(CustomTextureDesc {
            name: "custom_draw_video_luma_texture".to_string(),
            dimension: CustomTextureDimension::D2,
            width: frame.luma_width,
            height: frame.luma_height,
            format: CustomTextureFormat::R8Unorm,
            usage: CustomTextureUsage::SAMPLED,
            data: vec![frame.luma_data.clone()],
        })?;

        let chroma_texture = window.create_custom_texture(CustomTextureDesc {
            name: "custom_draw_video_chroma_texture".to_string(),
            dimension: CustomTextureDimension::D2,
            width: frame.chroma_width,
            height: frame.chroma_height,
            format: CustomTextureFormat::Rg8Unorm,
            usage: CustomTextureUsage::SAMPLED,
            data: vec![frame.chroma_data.clone()],
        })?;

        Ok((luma_texture, chroma_texture))
    }

    fn update_video_textures(
        window: &mut Window,
        resources: &mut VideoResources,
        frame: DecodedVideoFrame,
    ) -> anyhow::Result<()> {
        let frame_layout = VideoFrameLayout::from_frame(&frame);
        if frame_layout != resources.frame_layout {
            window
                .remove_custom_texture(resources.luma_texture)
                .map_err(|error| {
                    anyhow!("failed to remove old luma texture before resize: {error}")
                })?;
            window
                .remove_custom_texture(resources.chroma_texture)
                .map_err(|error| {
                    anyhow!("failed to remove old chroma texture before resize: {error}")
                })?;

            let (luma_texture, chroma_texture) = create_video_textures(window, &frame)?;
            resources.luma_texture = luma_texture;
            resources.chroma_texture = chroma_texture;
            resources.frame_layout = frame_layout;
            return Ok(());
        }

        window
            .update_custom_texture(
                resources.luma_texture,
                CustomTextureUpdate {
                    level: 0,
                    data: frame.luma_data,
                    bytes_per_row: Some(frame.luma_bytes_per_row),
                },
            )
            .map_err(|error| anyhow!("video luma texture upload failed: {error}"))?;

        window
            .update_custom_texture(
                resources.chroma_texture,
                CustomTextureUpdate {
                    level: 0,
                    data: frame.chroma_data,
                    bytes_per_row: Some(frame.chroma_bytes_per_row),
                },
            )
            .map_err(|error| anyhow!("video chroma texture upload failed: {error}"))?;

        Ok(())
    }

    fn format_clock_time(seconds: f64) -> String {
        if !seconds.is_finite() {
            return "00:00".to_string();
        }

        let total_seconds = seconds.max(0.0).round() as u64;
        let minutes = total_seconds / 60;
        let remaining_seconds = total_seconds % 60;
        format!("{minutes:02}:{remaining_seconds:02}")
    }

    fn append_f32(output: &mut Vec<u8>, value: f32) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn quad_vertex_data() -> Arc<[u8]> {
        let mut data = Vec::with_capacity(6 * 4 * 4);
        let vertices = [
            (-0.9, -0.9, 0.0, 1.0),
            (0.9, -0.9, 1.0, 1.0),
            (0.9, 0.9, 1.0, 0.0),
            (-0.9, -0.9, 0.0, 1.0),
            (0.9, 0.9, 1.0, 0.0),
            (-0.9, 0.9, 0.0, 0.0),
        ];

        for (x, y, u, v) in vertices {
            append_f32(&mut data, x);
            append_f32(&mut data, y);
            append_f32(&mut data, u);
            append_f32(&mut data, v);
        }

        Arc::from(data)
    }

    fn quad_vertex_data_for_bounds(
        bounds: Bounds<gpui::Pixels>,
        viewport: gpui::Size<gpui::Pixels>,
    ) -> Arc<[u8]> {
        let mut data = Vec::with_capacity(6 * 4 * 4);

        let left = bounds.origin.x;
        let top = bounds.origin.y;
        let right = bounds.origin.x + bounds.size.width;
        let bottom = bounds.origin.y + bounds.size.height;

        let viewport_width = f32::from(viewport.width).max(1.0);
        let viewport_height = f32::from(viewport.height).max(1.0);

        let ndc_x = |x: gpui::Pixels| (f32::from(x) / viewport_width) * 2.0 - 1.0;
        let ndc_y = |y: gpui::Pixels| 1.0 - (f32::from(y) / viewport_height) * 2.0;

        let left_ndc = ndc_x(left);
        let right_ndc = ndc_x(right);
        let top_ndc = ndc_y(top);
        let bottom_ndc = ndc_y(bottom);

        let vertices = [
            (left_ndc, top_ndc, 0.0, 0.0),
            (right_ndc, top_ndc, 1.0, 0.0),
            (right_ndc, bottom_ndc, 1.0, 1.0),
            (left_ndc, top_ndc, 0.0, 0.0),
            (right_ndc, bottom_ndc, 1.0, 1.0),
            (left_ndc, bottom_ndc, 0.0, 1.0),
        ];

        for (x, y, u, v) in vertices {
            append_f32(&mut data, x);
            append_f32(&mut data, y);
            append_f32(&mut data, u);
            append_f32(&mut data, v);
        }

        Arc::from(data)
    }

    pub fn run() {
        let config = VideoExampleConfig::from_args();
        application().run(move |cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(760.0), px(560.0)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| {
                    let config = config.clone();
                    cx.new(|cx| VideoCustomDrawExample::new(config, cx))
                },
            )
            .expect("failed to open custom draw video window");
        });
    }
}

#[cfg(feature = "video-ffmpeg")]
fn main() {
    enabled::run();
}
