#![cfg_attr(target_family = "wasm", no_main)]

use std::{
    cell::RefCell,
    f32::consts::{PI, TAU},
    rc::Rc,
    sync::Arc,
    time::Instant,
};

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use gpui::colors::Colors;
use gpui::{
    App, AppContext, Bounds, Context, CustomBindingDesc, CustomBindingKind, CustomBindingName,
    CustomBindingValue, CustomBlendMode, CustomBufferDesc, CustomBufferId, CustomBufferSource,
    CustomCullMode, CustomDepthCompare, CustomDepthFormat, CustomDepthState, CustomDrawParams,
    CustomIndexBuffer, CustomIndexFormat, CustomPipelineDesc, CustomPipelineId,
    CustomPipelineState, CustomPrimitiveTopology, CustomUniformBuilder, CustomVertexAttribute,
    CustomVertexAttributeName, CustomVertexBuffer, CustomVertexFetch, CustomVertexFormat,
    CustomVertexLayout, Hsla, MouseButton, Render, Styled, Window, WindowBounds, WindowOptions,
    canvas, div, prelude::*, px, size,
};
use gpui_platform::application;

const SURFACE_WIDTH: f32 = 760.0;
const SURFACE_HEIGHT: f32 = 520.0;
const UNIFORM_SIZE: u32 = 128;
const FIXED_SIMULATION_STEP_SECONDS: f32 = 1.0 / 120.0;
const MAX_SIMULATION_STEPS_PER_FRAME: usize = 8;
const BALL_RADIUS: f32 = 0.26;
const RACK_ROWS: usize = 5;
const TABLE_HALF_WIDTH: f32 = 7.8;
const TABLE_HALF_DEPTH: f32 = 4.3;
const TABLE_SURFACE_Y: f32 = 0.0;
const CUE_BALL_START_X: f32 = -4.6;
const RACK_APEX_X: f32 = 3.3;
const SHOT_MAX_DRAG_DISTANCE: f32 = 3.4;
const SHOT_MAX_SPEED: f32 = 24.0;
const SHOT_READY_SPEED_THRESHOLD: f32 = 0.22;
const MOTION_SLEEP_SPEED_THRESHOLD: f32 = 0.045;
const AIM_GUIDE_MARKER_COUNT: usize = 12;
const POCKET_RADIUS: f32 = 0.55;
const POCKET_VISUAL_RADIUS: f32 = 0.40;
const SHADOW_RADIUS: f32 = 0.20;
const TOTAL_OBJECT_BALLS: usize = 15;
const RAIL_MARKER_RADIUS: f32 = 0.06;
const RAIL_INSET: f32 = 0.25;

const SHADER_SOURCE: &str = r#"
struct VertexInput {
  a0: vec3<f32>,
  a1: vec3<f32>,
  a2: vec3<f32>,
  a3: f32,
  a4: vec4<f32>,
  a5: f32,
};

struct SceneUniforms {
  view_proj: mat4x4<f32>,
  bounds: vec4<f32>,
  viewport: vec4<f32>,
  camera_position: vec4<f32>,
  light_direction: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_position: vec3<f32>,
  @location(1) world_normal: vec3<f32>,
  @location(2) color: vec4<f32>,
};

var<uniform> b0: SceneUniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;

  let scaled = vec3<f32>(input.a0.x * input.a3, input.a0.y * input.a3 * input.a5, input.a0.z * input.a3);
  let world_position = input.a2 + scaled;
  let clip = b0.view_proj * vec4<f32>(world_position, 1.0);
  let local_ndc = clip.xyz / clip.w;

  let pixel = vec2<f32>(
    b0.bounds.x + (local_ndc.x * 0.5 + 0.5) * b0.bounds.z,
    b0.bounds.y + (1.0 - (local_ndc.y * 0.5 + 0.5)) * b0.bounds.w
  );

  let mapped_ndc = vec2<f32>(
    (pixel.x / b0.viewport.x) * 2.0 - 1.0,
    1.0 - (pixel.y / b0.viewport.y) * 2.0
  );

  out.position = vec4<f32>(mapped_ndc, local_ndc.z, 1.0);
  out.world_position = world_position;
  let y_scale_rcp = select(1.0 / input.a5, 1.0, input.a5 < 0.001);
  out.world_normal = normalize(vec3<f32>(input.a1.x, input.a1.y * y_scale_rcp, input.a1.z));
  out.color = input.a4;
  return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let normal = normalize(input.world_normal);
  let key_light_direction = normalize(b0.light_direction.xyz);
  let fill_light_direction = normalize(vec3<f32>(0.55, -0.3, 0.7));
  let back_light_direction = normalize(vec3<f32>(-0.25, -0.15, 1.0));

  let view_direction = normalize(b0.camera_position.xyz - input.world_position);

  let key_diffuse = max(dot(normal, -key_light_direction), 0.0);
  let fill_diffuse = max(dot(normal, -fill_light_direction), 0.0);
  let back_diffuse = max(dot(normal, -back_light_direction), 0.0);

  let key_half_direction = normalize(view_direction - key_light_direction);
  let key_specular = pow(max(dot(normal, key_half_direction), 0.0), 36.0);

  let fill_half_direction = normalize(view_direction - fill_light_direction);
  let fill_specular = pow(max(dot(normal, fill_half_direction), 0.0), 20.0);

  let hemisphere_mix = normal.y * 0.5 + 0.5;
  let sky_color = vec3<f32>(0.42, 0.46, 0.53);
  let ground_color = vec3<f32>(0.15, 0.16, 0.18);
  let hemisphere = ground_color * (1.0 - hemisphere_mix) + sky_color * hemisphere_mix;

  let rim = pow(1.0 - max(dot(normal, view_direction), 0.0), 2.4);

  let diffuse_lighting = key_diffuse * 0.95 + fill_diffuse * 0.42 + back_diffuse * 0.18;
  var lit_color = input.color.rgb * (hemisphere + diffuse_lighting);
  lit_color += vec3<f32>(1.0, 0.97, 0.92) * (key_specular * 0.24 + fill_specular * 0.11);
  lit_color += vec3<f32>(0.42, 0.58, 0.92) * rim * 0.12;
  lit_color = max(lit_color, vec3<f32>(0.0, 0.0, 0.0));

  return vec4<f32>(lit_color, 1.0);
}
"#;

type Mat4 = [f32; 16];

#[derive(Clone, Copy)]
struct CameraRig {
    eye: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
    fov_y_radians: f32,
    near_plane: f32,
    far_plane: f32,
}

#[derive(Component, Clone, Copy)]
struct BallBody {
    radius: f32,
    inverse_mass: f32,
}

#[derive(Component, Clone, Copy)]
struct WorldPosition {
    value: [f32; 3],
}

#[derive(Component, Clone, Copy)]
struct LinearVelocity {
    value: [f32; 3],
}

#[derive(Component, Clone, Copy)]
struct BallColor {
    value: [f32; 4],
}

#[derive(Component, Clone, Copy)]
struct CueBall;

#[derive(Resource, Clone, Copy)]
struct SimulationConfig {
    restitution: f32,
    linear_damping: f32,
    table_half_width: f32,
    table_half_depth: f32,
    table_surface_y: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            restitution: 0.91,
            linear_damping: 1.7,
            table_half_width: TABLE_HALF_WIDTH,
            table_half_depth: TABLE_HALF_DEPTH,
            table_surface_y: TABLE_SURFACE_Y,
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct SimulationStep {
    delta_seconds: f32,
}

struct BevyEcsCustomDrawExample {
    pipeline: Option<CustomPipelineId>,
    sphere_vertex_buffer: Option<CustomBufferId>,
    sphere_index_buffer: Option<CustomBufferId>,
    instance_buffer: Option<CustomBufferId>,
    sphere_vertex_count: u32,
    sphere_index_count: u32,
    simulation_world: World,
    simulation_schedule: Schedule,
    surface_draw_bounds: Rc<RefCell<Option<Bounds<gpui::Pixels>>>>,
    last_frame_instant: Instant,
    simulation_accumulator_seconds: f32,
    paused: bool,
    shot_drag_active: bool,
    shot_drag_target: Option<[f32; 3]>,
    pocketed_ball_colors: Vec<[f32; 4]>,
    error: Option<String>,
}

impl BevyEcsCustomDrawExample {
    fn new(_cx: &mut Context<Self>) -> Self {
        let (simulation_world, simulation_schedule) = create_simulation_world_and_schedule();
        Self {
            pipeline: None,
            sphere_vertex_buffer: None,
            sphere_index_buffer: None,
            instance_buffer: None,
            sphere_vertex_count: 0,
            sphere_index_count: 0,
            simulation_world,
            simulation_schedule,
            surface_draw_bounds: Rc::new(RefCell::new(None)),
            last_frame_instant: Instant::now(),
            simulation_accumulator_seconds: 0.0,
            paused: false,
            shot_drag_active: false,
            shot_drag_target: None,
            pocketed_ball_colors: Vec::new(),
            error: None,
        }
    }

    fn ensure_resources(&mut self, window: &mut Window) {
        if self.pipeline.is_some() || self.error.is_some() {
            return;
        }

        match self.build_resources(window) {
            Ok((
                pipeline,
                sphere_vertex_buffer,
                sphere_index_buffer,
                instance_buffer,
                sphere_vertex_count,
                sphere_index_count,
            )) => {
                self.pipeline = Some(pipeline);
                self.sphere_vertex_buffer = Some(sphere_vertex_buffer);
                self.sphere_index_buffer = Some(sphere_index_buffer);
                self.instance_buffer = Some(instance_buffer);
                self.sphere_vertex_count = sphere_vertex_count;
                self.sphere_index_count = sphere_index_count;
            }
            Err(error) => {
                self.error = Some(error.to_string());
            }
        }
    }

    fn build_resources(
        &mut self,
        window: &mut Window,
    ) -> anyhow::Result<(
        CustomPipelineId,
        CustomBufferId,
        CustomBufferId,
        CustomBufferId,
        u32,
        u32,
    )> {
        let pipeline = window.create_custom_pipeline(CustomPipelineDesc {
            name: "custom_draw_bevy_ecs".to_string(),
            shader_source: SHADER_SOURCE.to_string(),
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            vertex_fetches: vec![
                CustomVertexFetch {
                    layout: CustomVertexLayout {
                        stride: 24,
                        attributes: vec![
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A0,
                                offset: 0,
                                format: CustomVertexFormat::F32Vec3,
                                location: None,
                            },
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A1,
                                offset: 12,
                                format: CustomVertexFormat::F32Vec3,
                                location: None,
                            },
                        ],
                    },
                    instanced: false,
                },
                CustomVertexFetch {
                    layout: CustomVertexLayout {
                        stride: 36,
                        attributes: vec![
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A2,
                                offset: 0,
                                format: CustomVertexFormat::F32Vec3,
                                location: None,
                            },
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A3,
                                offset: 12,
                                format: CustomVertexFormat::F32,
                                location: None,
                            },
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A4,
                                offset: 16,
                                format: CustomVertexFormat::F32Vec4,
                                location: None,
                            },
                            CustomVertexAttribute {
                                name: CustomVertexAttributeName::A5,
                                offset: 32,
                                format: CustomVertexFormat::F32,
                                location: None,
                            },
                        ],
                    },
                    instanced: true,
                },
            ],
            primitive: CustomPrimitiveTopology::TriangleList,
            color_targets: Vec::new(),
            state: CustomPipelineState {
                blend: CustomBlendMode::Opaque,
                cull_mode: CustomCullMode::Back,
                depth: Some(CustomDepthState {
                    format: CustomDepthFormat::Depth32Float,
                    compare: CustomDepthCompare::LessEqual,
                    write_enabled: true,
                }),
                ..CustomPipelineState::default()
            },
            push_constants: None,
            bindings: vec![CustomBindingDesc {
                name: CustomBindingName::B0,
                kind: CustomBindingKind::Uniform { size: UNIFORM_SIZE },
                slot: None,
            }],
        })?;

        let (sphere_vertices, sphere_indices, sphere_vertex_count, sphere_index_count) =
            sphere_mesh_data(16, 24);

        let sphere_vertex_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "bevy_ecs_sphere_vertices".to_string(),
            data: sphere_vertices,
        })?;

        let sphere_index_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "bevy_ecs_sphere_indices".to_string(),
            data: sphere_indices,
        })?;

        let (instance_data, _) = self.instance_data();
        let instance_buffer = window.create_custom_buffer(CustomBufferDesc {
            name: "bevy_ecs_ball_instances".to_string(),
            data: instance_data,
        })?;

        Ok((
            pipeline,
            sphere_vertex_buffer,
            sphere_index_buffer,
            instance_buffer,
            sphere_vertex_count,
            sphere_index_count,
        ))
    }

    fn ball_count(&mut self) -> usize {
        let mut ball_query = self
            .simulation_world
            .query::<(&WorldPosition, &BallBody, &BallColor)>();
        ball_query.iter(&self.simulation_world).count()
    }

    fn instance_data(&mut self) -> (Arc<[u8]>, u32) {
        let mut instance_data = Vec::new();
        let mut instance_count = 0u32;
        let mut cue_ball_position = None;
        let mut ball_positions: Vec<[f32; 3]> = Vec::new();

        let mut ball_query =
            self.simulation_world
                .query::<(&WorldPosition, &BallBody, &BallColor, Option<&CueBall>)>();
        for (world_position, ball_body, ball_color, cue_ball_marker) in
            ball_query.iter(&self.simulation_world)
        {
            append_f32(&mut instance_data, world_position.value[0]);
            append_f32(&mut instance_data, world_position.value[1]);
            append_f32(&mut instance_data, world_position.value[2]);
            append_f32(&mut instance_data, ball_body.radius);
            append_f32(&mut instance_data, ball_color.value[0]);
            append_f32(&mut instance_data, ball_color.value[1]);
            append_f32(&mut instance_data, ball_color.value[2]);
            append_f32(&mut instance_data, ball_color.value[3]);
            append_f32(&mut instance_data, 1.0);
            instance_count = instance_count.saturating_add(1);
            ball_positions.push(world_position.value);

            if cue_ball_marker.is_some() {
                cue_ball_position = Some(world_position.value);
            }
        }

        for position in &ball_positions {
            append_f32(&mut instance_data, position[0]);
            append_f32(&mut instance_data, TABLE_SURFACE_Y + 0.005);
            append_f32(&mut instance_data, position[2]);
            append_f32(&mut instance_data, SHADOW_RADIUS);
            append_f32(&mut instance_data, 0.03);
            append_f32(&mut instance_data, 0.04);
            append_f32(&mut instance_data, 0.03);
            append_f32(&mut instance_data, 1.0);
            append_f32(&mut instance_data, 0.12);
            instance_count = instance_count.saturating_add(1);
        }

        for pocket in pocket_positions() {
            append_f32(&mut instance_data, pocket[0]);
            append_f32(&mut instance_data, pocket[1] + 0.01);
            append_f32(&mut instance_data, pocket[2]);
            append_f32(&mut instance_data, POCKET_VISUAL_RADIUS);
            append_f32(&mut instance_data, 0.04);
            append_f32(&mut instance_data, 0.04);
            append_f32(&mut instance_data, 0.05);
            append_f32(&mut instance_data, 1.0);
            append_f32(&mut instance_data, 0.03);
            instance_count = instance_count.saturating_add(1);
        }

        let rail_color: [f32; 3] = [0.28, 0.18, 0.10];
        let half_width = TABLE_HALF_WIDTH;
        let half_depth = TABLE_HALF_DEPTH;
        let rail_y = TABLE_SURFACE_Y + RAIL_MARKER_RADIUS;
        let long_count = ((half_width * 2.0) / 0.9) as usize;
        let short_count = ((half_depth * 2.0) / 0.9) as usize;
        for index in 0..=long_count {
            let ratio = index as f32 / long_count as f32;
            let x = -half_width + ratio * half_width * 2.0;
            for &z_sign in &[-1.0f32, 1.0] {
                let z = z_sign * (half_depth + RAIL_INSET * 0.2);
                append_f32(&mut instance_data, x);
                append_f32(&mut instance_data, rail_y);
                append_f32(&mut instance_data, z);
                append_f32(&mut instance_data, RAIL_MARKER_RADIUS);
                append_f32(&mut instance_data, rail_color[0]);
                append_f32(&mut instance_data, rail_color[1]);
                append_f32(&mut instance_data, rail_color[2]);
                append_f32(&mut instance_data, 1.0);
                append_f32(&mut instance_data, 1.0);
                instance_count = instance_count.saturating_add(1);
            }
        }
        for index in 1..short_count {
            let ratio = index as f32 / short_count as f32;
            let z = -half_depth + ratio * half_depth * 2.0;
            for &x_sign in &[-1.0f32, 1.0] {
                let x = x_sign * (half_width + RAIL_INSET * 0.2);
                append_f32(&mut instance_data, x);
                append_f32(&mut instance_data, rail_y);
                append_f32(&mut instance_data, z);
                append_f32(&mut instance_data, RAIL_MARKER_RADIUS);
                append_f32(&mut instance_data, rail_color[0]);
                append_f32(&mut instance_data, rail_color[1]);
                append_f32(&mut instance_data, rail_color[2]);
                append_f32(&mut instance_data, 1.0);
                append_f32(&mut instance_data, 1.0);
                instance_count = instance_count.saturating_add(1);
            }
        }

        if self.shot_drag_active
            && let (Some(cue_ball_position), Some(shot_drag_target)) =
                (cue_ball_position, self.shot_drag_target)
        {
            let drag_vector = subtract3(cue_ball_position, shot_drag_target);
            let horizontal_drag_vector = [drag_vector[0], 0.0, drag_vector[2]];
            let drag_distance = length3(horizontal_drag_vector);
            if drag_distance > 0.01 {
                let guide_direction = scale3(horizontal_drag_vector, drag_distance.recip());
                let guide_length = drag_distance.min(4.8);
                let power_ratio = (drag_distance / SHOT_MAX_DRAG_DISTANCE).min(1.0);
                let (guide_r, guide_g, guide_b) = if power_ratio < 0.5 {
                    let interpolation = power_ratio * 2.0;
                    (0.3 + interpolation * 0.7, 0.9, 0.3 - interpolation * 0.1)
                } else {
                    let interpolation = (power_ratio - 0.5) * 2.0;
                    (1.0, 0.9 - interpolation * 0.7, 0.2 - interpolation * 0.1)
                };

                for marker_index in 0..AIM_GUIDE_MARKER_COUNT {
                    let marker_ratio = (marker_index + 1) as f32 / AIM_GUIDE_MARKER_COUNT as f32;
                    let marker_position = add3(
                        cue_ball_position,
                        scale3(guide_direction, guide_length * marker_ratio),
                    );

                    append_f32(&mut instance_data, marker_position[0]);
                    append_f32(&mut instance_data, marker_position[1] + 0.04);
                    append_f32(&mut instance_data, marker_position[2]);
                    append_f32(&mut instance_data, 0.085 * (1.0 - marker_ratio * 0.45));
                    append_f32(&mut instance_data, guide_r);
                    append_f32(&mut instance_data, guide_g);
                    append_f32(&mut instance_data, guide_b);
                    append_f32(&mut instance_data, 1.0);
                    append_f32(&mut instance_data, 1.0);
                    instance_count = instance_count.saturating_add(1);
                }
            }
        }

        (Arc::from(instance_data), instance_count)
    }

    fn step_simulation(&mut self) {
        let now = Instant::now();
        let frame_delta_seconds = now
            .saturating_duration_since(self.last_frame_instant)
            .as_secs_f32()
            .min(0.1);
        self.last_frame_instant = now;

        if self.paused {
            return;
        }

        self.simulation_accumulator_seconds += frame_delta_seconds;
        let mut simulation_steps = 0usize;

        while self.simulation_accumulator_seconds >= FIXED_SIMULATION_STEP_SECONDS
            && simulation_steps < MAX_SIMULATION_STEPS_PER_FRAME
        {
            if let Some(mut simulation_step) =
                self.simulation_world.get_resource_mut::<SimulationStep>()
            {
                simulation_step.delta_seconds = FIXED_SIMULATION_STEP_SECONDS;
            }

            self.simulation_schedule.run(&mut self.simulation_world);
            self.simulation_accumulator_seconds -= FIXED_SIMULATION_STEP_SECONDS;
            simulation_steps = simulation_steps.saturating_add(1);
        }

        if simulation_steps == MAX_SIMULATION_STEPS_PER_FRAME {
            self.simulation_accumulator_seconds = 0.0;
        }

        self.process_pockets();
    }

    fn process_pockets(&mut self) {
        let pockets = pocket_positions();
        let mut to_despawn: Vec<(Entity, [f32; 4])> = Vec::new();
        let mut cue_ball_scratched = false;

        {
            let mut ball_query =
                self.simulation_world
                    .query::<(Entity, &WorldPosition, &BallColor, Option<&CueBall>)>();
            for (entity, position, color, cue_marker) in ball_query.iter(&self.simulation_world) {
                for pocket in &pockets {
                    let delta_x = position.value[0] - pocket[0];
                    let delta_z = position.value[2] - pocket[2];
                    if delta_x * delta_x + delta_z * delta_z < POCKET_RADIUS * POCKET_RADIUS {
                        if cue_marker.is_some() {
                            cue_ball_scratched = true;
                        } else {
                            to_despawn.push((entity, color.value));
                        }
                        break;
                    }
                }
            }
        }

        for (entity, color) in to_despawn {
            self.pocketed_ball_colors.push(color);
            self.simulation_world.entity_mut(entity).despawn();
        }

        if cue_ball_scratched {
            let cue_entity: Option<Entity> = {
                let mut query = self.simulation_world.query::<(Entity, &CueBall)>();
                query
                    .iter(&self.simulation_world)
                    .next()
                    .map(|(entity, _)| entity)
            };
            if let Some(entity) = cue_entity {
                let respawn_position = self.find_cue_ball_respawn_position(entity);
                if let Some(mut position) = self.simulation_world.get_mut::<WorldPosition>(entity) {
                    position.value = respawn_position;
                }
                if let Some(mut velocity) = self.simulation_world.get_mut::<LinearVelocity>(entity)
                {
                    velocity.value = [0.0, 0.0, 0.0];
                }
                self.shot_drag_active = false;
                self.shot_drag_target = None;
            }
        }
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    fn reset_simulation(&mut self) {
        let (simulation_world, simulation_schedule) = create_simulation_world_and_schedule();
        self.simulation_world = simulation_world;
        self.simulation_schedule = simulation_schedule;
        self.simulation_accumulator_seconds = 0.0;
        self.last_frame_instant = Instant::now();
        self.shot_drag_active = false;
        self.shot_drag_target = None;
        self.pocketed_ball_colors.clear();
    }

    fn find_cue_ball_respawn_position(&mut self, cue_entity: Entity) -> [f32; 3] {
        let (table_half_width, table_half_depth, table_surface_y) = if let Some(simulation_config) =
            self.simulation_world
                .get_resource::<SimulationConfig>()
                .copied()
        {
            (
                simulation_config.table_half_width,
                simulation_config.table_half_depth,
                simulation_config.table_surface_y,
            )
        } else {
            (TABLE_HALF_WIDTH, TABLE_HALF_DEPTH, TABLE_SURFACE_Y)
        };

        let center_y = table_surface_y + BALL_RADIUS;
        let min_x = -table_half_width + BALL_RADIUS;
        let max_x = table_half_width - BALL_RADIUS;
        let min_z = -table_half_depth + BALL_RADIUS;
        let max_z = table_half_depth - BALL_RADIUS;

        let start_x = CUE_BALL_START_X.clamp(min_x, max_x);
        let z_step = BALL_RADIUS * 2.25;
        let z_offsets = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0];

        for lane_index in 0..8 {
            let candidate_x = (start_x + lane_index as f32 * BALL_RADIUS * 1.5).clamp(min_x, max_x);
            for z_offset in z_offsets {
                let candidate_z = (z_offset * z_step).clamp(min_z, max_z);
                let candidate_position = [candidate_x, center_y, candidate_z];
                if self.cue_spawn_position_is_clear(cue_entity, candidate_position) {
                    return candidate_position;
                }
            }
        }

        [start_x, center_y, 0.0f32.clamp(min_z, max_z)]
    }

    fn cue_spawn_position_is_clear(
        &mut self,
        cue_entity: Entity,
        candidate_position: [f32; 3],
    ) -> bool {
        let mut ball_query = self
            .simulation_world
            .query::<(Entity, &WorldPosition, &BallBody)>();
        for (entity, world_position, ball_body) in ball_query.iter(&self.simulation_world) {
            if entity == cue_entity {
                continue;
            }

            let minimum_clearance = BALL_RADIUS + ball_body.radius + 0.04;
            if horizontal_distance(world_position.value, candidate_position) < minimum_clearance {
                return false;
            }
        }

        true
    }

    fn pointer_ray_for_surface_position(
        &self,
        pointer_position: gpui::Point<gpui::Pixels>,
    ) -> Option<([f32; 3], [f32; 3])> {
        let draw_bounds = {
            let draw_bounds_state = self.surface_draw_bounds.borrow();
            *draw_bounds_state
        }?;

        if !draw_bounds.contains(&pointer_position) {
            return None;
        }

        let local_x = f32::from(pointer_position.x - draw_bounds.origin.x);
        let local_y = f32::from(pointer_position.y - draw_bounds.origin.y);
        let draw_width = f32::from(draw_bounds.size.width).max(1.0);
        let draw_height = f32::from(draw_bounds.size.height).max(1.0);

        let camera_rig = default_camera_rig();
        let (ray_origin, ray_direction) =
            camera_ray_for_surface_point(local_x, local_y, draw_width, draw_height, camera_rig);
        Some((ray_origin, ray_direction))
    }

    fn pointer_world_on_table(
        &self,
        pointer_position: gpui::Point<gpui::Pixels>,
    ) -> Option<[f32; 3]> {
        let (ray_origin, ray_direction) =
            self.pointer_ray_for_surface_position(pointer_position)?;
        let simulation_config = self
            .simulation_world
            .get_resource::<SimulationConfig>()
            .copied()?;

        if ray_direction[1].abs() <= 1e-5 {
            return None;
        }

        let center_y = simulation_config.table_surface_y + BALL_RADIUS;
        let ray_parameter = (center_y - ray_origin[1]) / ray_direction[1];
        if ray_parameter <= 0.0 {
            return None;
        }

        let hit_position = add3(ray_origin, scale3(ray_direction, ray_parameter));
        Some([hit_position[0], center_y, hit_position[2]])
    }

    fn cue_ball_state(&mut self) -> Option<(Entity, [f32; 3], [f32; 3], f32)> {
        let mut cue_ball_query =
            self.simulation_world
                .query::<(Entity, &CueBall, &WorldPosition, &LinearVelocity, &BallBody)>();
        cue_ball_query.iter(&self.simulation_world).next().map(
            |(entity, _cue_ball, world_position, linear_velocity, ball_body)| {
                (
                    entity,
                    world_position.value,
                    linear_velocity.value,
                    ball_body.radius,
                )
            },
        )
    }

    fn cue_ball_speed(&mut self) -> f32 {
        self.cue_ball_state()
            .map_or(0.0, |(_, _, linear_velocity, _)| {
                horizontal_speed(linear_velocity)
            })
    }

    fn begin_shot_drag(&mut self, pointer_position: gpui::Point<gpui::Pixels>) {
        if self.paused {
            return;
        }

        let Some(pointer_world_position) = self.pointer_world_on_table(pointer_position) else {
            return;
        };
        let Some((_cue_ball_entity, cue_ball_position, cue_ball_velocity, cue_ball_radius)) =
            self.cue_ball_state()
        else {
            return;
        };

        if horizontal_speed(cue_ball_velocity) > SHOT_READY_SPEED_THRESHOLD {
            return;
        }

        let pointer_distance_to_cue =
            horizontal_distance(pointer_world_position, cue_ball_position);
        if pointer_distance_to_cue > cue_ball_radius * 2.6 {
            return;
        }

        self.shot_drag_active = true;
        self.shot_drag_target = Some(pointer_world_position);
    }

    fn update_shot_drag(&mut self, pointer_position: gpui::Point<gpui::Pixels>) {
        if !self.shot_drag_active {
            return;
        }

        if let Some(pointer_world_position) = self.pointer_world_on_table(pointer_position) {
            self.shot_drag_target = Some(pointer_world_position);
        }
    }

    fn release_shot_drag(&mut self, pointer_position: gpui::Point<gpui::Pixels>) {
        if !self.shot_drag_active {
            return;
        }

        if let Some(pointer_world_position) = self.pointer_world_on_table(pointer_position) {
            self.shot_drag_target = Some(pointer_world_position);
        }

        let shot_target = self.shot_drag_target;
        self.shot_drag_active = false;
        self.shot_drag_target = None;

        if self.paused {
            return;
        }

        let Some(shot_target) = shot_target else {
            return;
        };
        let Some((cue_ball_entity, cue_ball_position, cue_ball_velocity, _cue_ball_radius)) =
            self.cue_ball_state()
        else {
            return;
        };

        if horizontal_speed(cue_ball_velocity) > SHOT_READY_SPEED_THRESHOLD {
            return;
        }

        let raw_drag_vector = subtract3(cue_ball_position, shot_target);
        let horizontal_drag_vector = [raw_drag_vector[0], 0.0, raw_drag_vector[2]];
        let drag_distance = length3(horizontal_drag_vector);
        if drag_distance < 0.06 {
            return;
        }

        let clamped_drag_distance = drag_distance.min(SHOT_MAX_DRAG_DISTANCE);
        let shot_direction = scale3(horizontal_drag_vector, drag_distance.recip());
        let shot_speed = (clamped_drag_distance / SHOT_MAX_DRAG_DISTANCE) * SHOT_MAX_SPEED;

        if let Some(mut cue_ball_velocity) = self
            .simulation_world
            .get_mut::<LinearVelocity>(cue_ball_entity)
        {
            cue_ball_velocity.value =
                add3(cue_ball_velocity.value, scale3(shot_direction, shot_speed));
            cue_ball_velocity.value[1] = 0.0;
        }
    }
}

impl Render for BevyEcsCustomDrawExample {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let colors = Colors::for_appearance(window);
        self.ensure_resources(window);
        if self.error.is_none() {
            self.step_simulation();
        }
        window.request_animation_frame();

        let ball_count = self.ball_count();
        let pocketed_count = self.pocketed_ball_colors.len();
        let all_pocketed = pocketed_count >= TOTAL_OBJECT_BALLS;
        let cue_ball_speed = self.cue_ball_speed();
        let shot_status_text = if all_pocketed {
            "All balls pocketed! Press Re-rack to play again."
        } else if self.shot_drag_active {
            "Drag to set power and release to shoot"
        } else if cue_ball_speed > SHOT_READY_SPEED_THRESHOLD {
            "Waiting for balls to settle..."
        } else {
            "Drag from the cue ball to aim and shoot"
        };

        let controls = div()
            .flex()
            .gap_2()
            .child(
                div()
                    .id("bevy-ecs-toggle-pause")
                    .px_3()
                    .py_1()
                    .rounded_md()
                    .cursor_pointer()
                    .text_color(colors.selected_text)
                    .bg(colors.selected)
                    .hover(|style| style.bg(colors.selected))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _: &gpui::MouseDownEvent, _, _| {
                            this.toggle_pause();
                        }),
                    )
                    .child(if self.paused { "Resume" } else { "Pause" }),
            )
            .child(
                div()
                    .id("bevy-ecs-reset")
                    .px_3()
                    .py_1()
                    .rounded_md()
                    .cursor_pointer()
                    .text_color(colors.selected_text)
                    .bg(colors.selected)
                    .hover(|style| style.bg(colors.selected))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _: &gpui::MouseDownEvent, _, _| {
                            this.reset_simulation();
                        }),
                    )
                    .child("Re-rack"),
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
                    .child("Bevy ECS Billiards"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(colors.disabled)
                    .child(shot_status_text),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(colors.disabled)
                    .child(format!(
                        "Pocketed: {pocketed_count}/{TOTAL_OBJECT_BALLS}  •  On table: {ball_count}  •  Cue speed: {cue_ball_speed:.2}"
                    )),
            )
            .when(all_pocketed, |header| {
                header.child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::BOLD)
                        .text_color(gpui::green())
                        .child("🎱 You cleared the table!"),
                )
            })
            .child(controls);

        let surface_color: Hsla = colors.container.into();
        let content = if let Some(error) = &self.error {
            div()
                .text_sm()
                .text_color(gpui::red())
                .child(format!("Custom draw unsupported: {error}"))
        } else if let (
            Some(pipeline),
            Some(sphere_vertex_buffer),
            Some(sphere_index_buffer),
            Some(instance_buffer),
        ) = (
            self.pipeline,
            self.sphere_vertex_buffer,
            self.sphere_index_buffer,
            self.instance_buffer,
        ) {
            let sphere_vertex_count = self.sphere_vertex_count;
            let sphere_index_count = self.sphere_index_count;
            let (instance_data, instance_count) = self.instance_data();
            let surface_draw_bounds = self.surface_draw_bounds.clone();
            let camera_rig = default_camera_rig();

            let prepaint = move |bounds: Bounds<_>, window: &mut Window, _cx: &mut App| {
                let draw_bounds = inset_bounds(bounds, px(1.0));
                *surface_draw_bounds.borrow_mut() = Some(draw_bounds);

                if let Err(error) =
                    window.update_custom_buffer(instance_buffer, Arc::clone(&instance_data))
                {
                    log::error!("bevy ecs instance buffer update failed: {error}");
                }

                let uniform_data =
                    build_scene_uniform(draw_bounds, window.viewport_size(), camera_rig);

                CustomDrawParams {
                    bounds: draw_bounds,
                    pipeline,
                    vertex_buffers: vec![
                        CustomVertexBuffer {
                            source: CustomBufferSource::Buffer(sphere_vertex_buffer),
                        },
                        CustomVertexBuffer {
                            source: CustomBufferSource::Buffer(instance_buffer),
                        },
                    ],
                    vertex_count: sphere_vertex_count,
                    index_buffer: Some(CustomIndexBuffer {
                        source: CustomBufferSource::Buffer(sphere_index_buffer),
                        format: CustomIndexFormat::U16,
                    }),
                    index_count: sphere_index_count,
                    target: None,
                    instance_count,
                    push_constants: None,
                    bindings: vec![CustomBindingValue::Uniform(CustomBufferSource::Inline(
                        uniform_data,
                    ))],
                }
            };

            let paint = move |_bounds: Bounds<_>,
                              params: CustomDrawParams,
                              window: &mut Window,
                              _cx: &mut App| {
                if let Err(error) = window.paint_custom(params) {
                    log::error!("bevy ecs custom draw paint failed: {error}");
                }
            };

            div()
                .w(px(SURFACE_WIDTH))
                .h(px(SURFACE_HEIGHT))
                .rounded_md()
                .border_1()
                .border_color(colors.border)
                .bg(surface_color.opacity(0.2))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, event: &gpui::MouseDownEvent, _, _| {
                        this.begin_shot_drag(event.position);
                    }),
                )
                .on_mouse_move(cx.listener(|this, event: &gpui::MouseMoveEvent, _, _| {
                    this.update_shot_drag(event.position);
                }))
                .child(canvas(prepaint, paint).size_full())
        } else {
            div()
                .text_sm()
                .text_color(colors.disabled)
                .child("Initializing custom draw resources...")
        };

        div()
            .size_full()
            .p_6()
            .bg(colors.background)
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseUpEvent, _, _| {
                    this.release_shot_drag(event.position);
                }),
            )
            .child(div().flex().flex_col().gap_4().child(header).child(content))
    }
}

fn create_simulation_world_and_schedule() -> (World, Schedule) {
    let mut simulation_world = World::new();
    simulation_world.insert_resource(SimulationConfig::default());
    simulation_world.insert_resource(SimulationStep { delta_seconds: 0.0 });

    let cue_ball_center_y = TABLE_SURFACE_Y + BALL_RADIUS;
    simulation_world.spawn((
        CueBall,
        BallBody {
            radius: BALL_RADIUS,
            inverse_mass: 1.0,
        },
        WorldPosition {
            value: [CUE_BALL_START_X, cue_ball_center_y, 0.0],
        },
        LinearVelocity {
            value: [0.0, 0.0, 0.0],
        },
        BallColor {
            value: [0.96, 0.96, 0.98, 1.0],
        },
    ));

    let rack_spacing = BALL_RADIUS * 2.08;
    let row_spacing = rack_spacing * 0.866_025_4;
    let mut rack_ball_index = 0usize;
    for row in 0..RACK_ROWS {
        let center_x = RACK_APEX_X + row as f32 * row_spacing;
        let row_start_z = -(row as f32) * rack_spacing * 0.5;
        for column in 0..=row {
            let center_z = row_start_z + column as f32 * rack_spacing;
            simulation_world.spawn((
                BallBody {
                    radius: BALL_RADIUS,
                    inverse_mass: 1.0,
                },
                WorldPosition {
                    value: [center_x, cue_ball_center_y, center_z],
                },
                LinearVelocity {
                    value: [0.0, 0.0, 0.0],
                },
                BallColor {
                    value: billiard_ball_color(rack_ball_index),
                },
            ));
            rack_ball_index = rack_ball_index.saturating_add(1);
        }
    }

    let mut simulation_schedule = Schedule::default();
    simulation_schedule.add_systems(
        (
            integrate_motion_system,
            solve_bounds_collisions_system,
            solve_ball_collisions_system,
        )
            .chain(),
    );

    (simulation_world, simulation_schedule)
}

fn integrate_motion_system(
    mut ball_query: Query<(&mut WorldPosition, &mut LinearVelocity, &BallBody)>,
    simulation_config: Res<SimulationConfig>,
    simulation_step: Res<SimulationStep>,
) {
    let delta_seconds = simulation_step.delta_seconds;
    if delta_seconds <= 0.0 {
        return;
    }

    let damping_factor = (1.0 - simulation_config.linear_damping * delta_seconds).max(0.0);

    for (mut world_position, mut linear_velocity, ball_body) in &mut ball_query {
        linear_velocity.value[1] = 0.0;
        world_position.value = add3(
            world_position.value,
            scale3(linear_velocity.value, delta_seconds),
        );
        linear_velocity.value = scale3(linear_velocity.value, damping_factor);

        if horizontal_speed(linear_velocity.value) <= MOTION_SLEEP_SPEED_THRESHOLD {
            linear_velocity.value = [0.0, 0.0, 0.0];
        }

        world_position.value[1] = simulation_config.table_surface_y + ball_body.radius;
    }
}

fn solve_bounds_collisions_system(
    mut ball_query: Query<(&mut WorldPosition, &mut LinearVelocity, &BallBody)>,
    simulation_config: Res<SimulationConfig>,
) {
    for (mut world_position, mut linear_velocity, ball_body) in &mut ball_query {
        let max_x = (simulation_config.table_half_width - ball_body.radius).max(ball_body.radius);
        let max_z = (simulation_config.table_half_depth - ball_body.radius).max(ball_body.radius);

        if world_position.value[0] < -max_x {
            world_position.value[0] = -max_x;
            if linear_velocity.value[0] < 0.0 {
                linear_velocity.value[0] =
                    -linear_velocity.value[0] * simulation_config.restitution;
                linear_velocity.value[2] *= 0.985;
            }
        } else if world_position.value[0] > max_x {
            world_position.value[0] = max_x;
            if linear_velocity.value[0] > 0.0 {
                linear_velocity.value[0] =
                    -linear_velocity.value[0] * simulation_config.restitution;
                linear_velocity.value[2] *= 0.985;
            }
        }

        if world_position.value[2] < -max_z {
            world_position.value[2] = -max_z;
            if linear_velocity.value[2] < 0.0 {
                linear_velocity.value[2] =
                    -linear_velocity.value[2] * simulation_config.restitution;
                linear_velocity.value[0] *= 0.985;
            }
        } else if world_position.value[2] > max_z {
            world_position.value[2] = max_z;
            if linear_velocity.value[2] > 0.0 {
                linear_velocity.value[2] =
                    -linear_velocity.value[2] * simulation_config.restitution;
                linear_velocity.value[0] *= 0.985;
            }
        }

        world_position.value[1] = simulation_config.table_surface_y + ball_body.radius;
        linear_velocity.value[1] = 0.0;
    }
}

fn solve_ball_collisions_system(
    mut ball_query: Query<(&mut WorldPosition, &mut LinearVelocity, &BallBody)>,
    simulation_config: Res<SimulationConfig>,
) {
    let mut combinations = ball_query.iter_combinations_mut();
    while let Some(
        [
            (mut first_position, mut first_velocity, first_body),
            (mut second_position, mut second_velocity, second_body),
        ],
    ) = combinations.fetch_next()
    {
        let center_offset = subtract3(second_position.value, first_position.value);
        let center_distance = length3(center_offset);
        let min_distance = first_body.radius + second_body.radius;
        if center_distance >= min_distance {
            continue;
        }

        let collision_normal = if center_distance > 1e-5 {
            scale3(center_offset, center_distance.recip())
        } else {
            [1.0, 0.0, 0.0]
        };

        let total_inverse_mass = first_body.inverse_mass + second_body.inverse_mass;
        if total_inverse_mass <= f32::EPSILON {
            continue;
        }

        let penetration = min_distance - center_distance;
        let first_correction = penetration * (first_body.inverse_mass / total_inverse_mass);
        let second_correction = penetration * (second_body.inverse_mass / total_inverse_mass);

        first_position.value = subtract3(
            first_position.value,
            scale3(collision_normal, first_correction),
        );
        second_position.value = add3(
            second_position.value,
            scale3(collision_normal, second_correction),
        );

        let relative_velocity = subtract3(second_velocity.value, first_velocity.value);
        let separating_speed = dot_product3(relative_velocity, collision_normal);
        if separating_speed >= 0.0 {
            continue;
        }

        let impulse_magnitude =
            -(1.0 + simulation_config.restitution) * separating_speed / total_inverse_mass;
        let impulse = scale3(collision_normal, impulse_magnitude);

        first_velocity.value = subtract3(
            first_velocity.value,
            scale3(impulse, first_body.inverse_mass),
        );
        second_velocity.value = add3(
            second_velocity.value,
            scale3(impulse, second_body.inverse_mass),
        );
    }
}

fn build_scene_uniform(
    bounds: Bounds<gpui::Pixels>,
    viewport: gpui::Size<gpui::Pixels>,
    camera_rig: CameraRig,
) -> Arc<[u8]> {
    let aspect = (f32::from(bounds.size.width) / f32::from(bounds.size.height).max(1.0)).max(0.2);
    let view = mat4_look_at(camera_rig.eye, camera_rig.target, camera_rig.up);
    let projection = mat4_perspective_rh_zo(
        camera_rig.fov_y_radians,
        aspect,
        camera_rig.near_plane,
        camera_rig.far_plane,
    );
    let view_projection = mat4_mul(projection, view);

    let light_direction = normalize3([-0.45, -1.0, -0.35]);
    let mut uniform_builder = CustomUniformBuilder::new();
    uniform_builder
        .push_mat4(view_projection)
        .push_vec4(
            f32::from(bounds.origin.x),
            f32::from(bounds.origin.y),
            f32::from(bounds.size.width),
            f32::from(bounds.size.height),
        )
        .push_vec4(
            f32::from(viewport.width).max(1.0),
            f32::from(viewport.height).max(1.0),
            0.0,
            0.0,
        )
        .push_vec4(camera_rig.eye[0], camera_rig.eye[1], camera_rig.eye[2], 0.0)
        .push_vec4(
            light_direction[0],
            light_direction[1],
            light_direction[2],
            0.0,
        );
    uniform_builder.finish()
}

fn sphere_mesh_data(
    latitude_segments: u32,
    longitude_segments: u32,
) -> (Arc<[u8]>, Arc<[u8]>, u32, u32) {
    let latitude_segments = latitude_segments.max(3);
    let longitude_segments = longitude_segments.max(3);

    let mut vertex_bytes = Vec::new();
    for latitude_index in 0..=latitude_segments {
        let latitude_ratio = latitude_index as f32 / latitude_segments as f32;
        let theta = latitude_ratio * PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for longitude_index in 0..=longitude_segments {
            let longitude_ratio = longitude_index as f32 / longitude_segments as f32;
            let phi = longitude_ratio * TAU;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = cos_phi * sin_theta;
            let y = cos_theta;
            let z = sin_phi * sin_theta;

            append_f32(&mut vertex_bytes, x);
            append_f32(&mut vertex_bytes, y);
            append_f32(&mut vertex_bytes, z);
            append_f32(&mut vertex_bytes, x);
            append_f32(&mut vertex_bytes, y);
            append_f32(&mut vertex_bytes, z);
        }
    }

    let row_stride = longitude_segments + 1;
    let mut index_bytes = Vec::new();
    for latitude_index in 0..latitude_segments {
        for longitude_index in 0..longitude_segments {
            let top_left = latitude_index * row_stride + longitude_index;
            let top_right = top_left + 1;
            let bottom_left = (latitude_index + 1) * row_stride + longitude_index;
            let bottom_right = bottom_left + 1;

            if latitude_index != 0 {
                append_u16(&mut index_bytes, top_left as u16);
                append_u16(&mut index_bytes, bottom_left as u16);
                append_u16(&mut index_bytes, top_right as u16);
            }

            if latitude_index != latitude_segments - 1 {
                append_u16(&mut index_bytes, top_right as u16);
                append_u16(&mut index_bytes, bottom_left as u16);
                append_u16(&mut index_bytes, bottom_right as u16);
            }
        }
    }

    let vertex_count = (latitude_segments + 1) * (longitude_segments + 1);
    let index_count = (index_bytes.len() / 2) as u32;
    (
        Arc::from(vertex_bytes),
        Arc::from(index_bytes),
        vertex_count,
        index_count,
    )
}

fn camera_ray_for_surface_point(
    local_x: f32,
    local_y: f32,
    surface_width: f32,
    surface_height: f32,
    camera_rig: CameraRig,
) -> ([f32; 3], [f32; 3]) {
    let normalized_x = (local_x / surface_width) * 2.0 - 1.0;
    let normalized_y = 1.0 - (local_y / surface_height) * 2.0;
    let aspect = (surface_width / surface_height.max(1.0)).max(0.2);
    let tan_half_fov = (camera_rig.fov_y_radians * 0.5).tan();

    let direction_camera = [
        normalized_x * aspect * tan_half_fov,
        normalized_y * tan_half_fov,
        -1.0,
    ];

    let camera_forward = normalize3(subtract3(camera_rig.target, camera_rig.eye));
    let camera_right = normalize3(cross_product3(camera_forward, camera_rig.up));
    let camera_up = cross_product3(camera_right, camera_forward);

    let world_direction = normalize3(add3(
        add3(
            scale3(camera_right, direction_camera[0]),
            scale3(camera_up, direction_camera[1]),
        ),
        scale3(camera_forward, -direction_camera[2]),
    ));

    (camera_rig.eye, world_direction)
}

fn pocket_positions() -> [[f32; 3]; 6] {
    let w = TABLE_HALF_WIDTH - RAIL_INSET;
    let d = TABLE_HALF_DEPTH - RAIL_INSET;
    let y = TABLE_SURFACE_Y;
    [
        [-w, y, -d],
        [-w, y, d],
        [w, y, -d],
        [w, y, d],
        [0.0, y, -d - RAIL_INSET * 0.5],
        [0.0, y, d + RAIL_INSET * 0.5],
    ]
}

fn default_camera_rig() -> CameraRig {
    CameraRig {
        eye: [0.0, 6.8, 15.6],
        target: [0.0, 1.7, 0.0],
        up: [0.0, 1.0, 0.0],
        fov_y_radians: 50.0f32.to_radians(),
        near_plane: 0.1,
        far_plane: 110.0,
    }
}

fn inset_bounds(bounds: Bounds<gpui::Pixels>, inset: gpui::Pixels) -> Bounds<gpui::Pixels> {
    let width = (bounds.size.width - inset * 2.0).max(px(1.0));
    let height = (bounds.size.height - inset * 2.0).max(px(1.0));
    Bounds {
        origin: bounds.origin + gpui::Point::new(inset, inset),
        size: gpui::Size::new(width, height),
    }
}

fn append_f32(output: &mut Vec<u8>, value: f32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn append_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn horizontal_speed(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[2] * vector[2]).sqrt()
}

fn horizontal_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    let delta_x = left[0] - right[0];
    let delta_z = left[2] - right[2];
    (delta_x * delta_x + delta_z * delta_z).sqrt()
}

fn billiard_ball_color(index: usize) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 8] = [
        [0.93, 0.83, 0.20, 1.0],
        [0.24, 0.46, 0.92, 1.0],
        [0.88, 0.18, 0.20, 1.0],
        [0.52, 0.25, 0.76, 1.0],
        [0.97, 0.52, 0.17, 1.0],
        [0.18, 0.63, 0.34, 1.0],
        [0.59, 0.21, 0.16, 1.0],
        [0.11, 0.11, 0.13, 1.0],
    ];

    PALETTE[index % PALETTE.len()]
}

fn add3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale3(vector: [f32; 3], scalar: f32) -> [f32; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn dot_product3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross_product3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn length3(vector: [f32; 3]) -> f32 {
    dot_product3(vector, vector).sqrt()
}

fn normalize3(vector: [f32; 3]) -> [f32; 3] {
    let vector_length = length3(vector);
    if vector_length <= f32::EPSILON {
        return [0.0, 1.0, 0.0];
    }
    scale3(vector, vector_length.recip())
}

fn mat4_mul(left: Mat4, right: Mat4) -> Mat4 {
    let mut result = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            result[column * 4 + row] = left[row] * right[column * 4]
                + left[4 + row] * right[column * 4 + 1]
                + left[8 + row] * right[column * 4 + 2]
                + left[12 + row] * right[column * 4 + 3];
        }
    }
    result
}

fn mat4_perspective_rh_zo(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let focal_length = 1.0 / (fov_y_radians * 0.5).tan();
    [
        focal_length / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        focal_length,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        (near * far) / (near - far),
        0.0,
    ]
}

fn mat4_look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let forward = normalize3(subtract3(target, eye));
    let side = normalize3(cross_product3(forward, up));
    let up_direction = cross_product3(side, forward);

    [
        side[0],
        up_direction[0],
        -forward[0],
        0.0,
        side[1],
        up_direction[1],
        -forward[1],
        0.0,
        side[2],
        up_direction[2],
        -forward[2],
        0.0,
        -dot_product3(side, eye),
        -dot_product3(up_direction, eye),
        dot_product3(forward, eye),
        1.0,
    ]
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(860.0), px(700.0)), cx);
        if let Err(error) = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(BevyEcsCustomDrawExample::new),
        ) {
            log::error!("failed to open bevy ecs custom draw example: {error}");
            return;
        }

        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    gpui_platform::web_init();
    run_example();
}
