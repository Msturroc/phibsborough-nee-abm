# phibsborough_ecs.jl — Ark.jl ECS conversion of Phibsborough traffic model
# Mirrors phibsborough_model_12.jl with identical API and logic.
# Uses StructArray (SoA) storage for cache-friendly iteration.

using Ark
using Random
using Distributions

# ============================================================================
# SECTION 1: CONSTANTS
# ============================================================================

const REGEN_EFF_ECS = try parse(Float64, get(ENV, "REGEN_EFF", "0.80")) catch; 0.80 end

# Direction codes
const DIR_UP    = 0x01
const DIR_DOWN  = 0x02
const DIR_LEFT  = 0x03
const DIR_RIGHT = 0x04

# Vehicle types
const VT_SMART_CAR = 0x01
const VT_CAR       = 0x02
const VT_E_CAR     = 0x03
const VT_SUV       = 0x04
const VT_E_SUV     = 0x05
const VT_BUS       = 0x06
const VT_OBSERVER  = 0x07  # Dedicated moving observer — physically identical to e-SUV

# Turning states
const TS_NOT_TURNING = 0x00
const TS_TURNING     = 0x01

# Turn intentions
const TI_STRAIGHT = 0x00
const TI_LEFT     = 0x01

# Traffic light states
const TL_GREEN = 0x01
const TL_AMBER = 0x02
const TL_RED   = 0x03

# Light types
const LT_HORIZONTAL = 0x01
const LT_VERTICAL   = 0x02

# Physics constants
const MAX_ACCELERATION_ECS = Float32(0.1)
const REACTION_DIST_ECS = Uniform(0.7, 1.3)

# Conversion tables (API boundary only)
const SYM_TO_VTYPE = Dict{Symbol,UInt8}(
    :smart_car => VT_SMART_CAR, :car => VT_CAR, :e_car => VT_E_CAR,
    :suv => VT_SUV, :e_suv => VT_E_SUV, :bus => VT_BUS,
    :observer => VT_OBSERVER)
const VTYPE_TO_SYM = Dict{UInt8,Symbol}(v => k for (k, v) in SYM_TO_VTYPE)
const SYM_TO_DIR = Dict{Symbol,UInt8}(:up => DIR_UP, :down => DIR_DOWN, :left => DIR_LEFT, :right => DIR_RIGHT)
const DIR_TO_SYM = Dict{UInt8,Symbol}(v => k for (k, v) in SYM_TO_DIR)

# Default traffic light timing
const DEFAULT_GREEN_S = 25.0
const DEFAULT_AMBER_S = 3.0
const DEFAULT_RED_S   = 32.0
const DEFAULT_CLEARANCE_S = 5.0

# Gaussian decay lookup tables (same as original)
const GAUSSIAN_DECAY_TABLE_ECS = let
    table = Dict{Int, Matrix{Float64}}()
    for r in 1:6
        side = 2r + 1
        m = zeros(side, side)
        rsq_div4 = r^2 / 4.0
        for xo in -r:r, yo in -r:r
            if xo == 0 && yo == 0; continue; end
            m[xo + r + 1, yo + r + 1] = exp(-(xo^2 + yo^2) / rsq_div4)
        end
        table[r] = m
    end
    table
end

# ============================================================================
# SECTION 2: COMPONENTS
# ============================================================================

struct GridPos
    x::Int32
    y::Int32
end

struct FracPos
    x::Float64
    y::Float64
end

struct VehicleId
    vehicle_type::UInt8
    direction::UInt8
    weight::Float32
    planning_distance::Int16
    traffic_light_idx::UInt32
end

struct VehicleMotion
    speed::Float32
    original_speed::Float32
    previous_speed::Float32
end

struct TurnState
    turning_state::UInt8
    turn_intention::UInt8
    turn_progress::Float32
    in_intersection::Bool
end

struct TrafficLightComp
    state::UInt8
    light_type::UInt8
    time_counter::Int32
end

# ============================================================================
# SECTION 3: SimState RESOURCE
# ============================================================================

mutable struct SimState
    # Grids
    brake_emissions::Matrix{Float64}
    tyre_emissions::Matrix{Float64}
    buildings::Matrix{Bool}
    occupancy_grid::Matrix{UInt32}

    # Entity mapping
    id_to_entity::Vector{Ark.Entity}

    # Intended moves (indexed by entity._id)
    intended_moves::Vector{NTuple{2,Int32}}
    intended_valid::BitVector
    intended_vehicle_type::Vector{UInt8}
    intended_direction::Vector{UInt8}

    # Simulation parameters
    dims::Tuple{Int,Int}
    decay_rate::Float64
    brake_decay_rate::Float64
    tyre_decay_rate::Float64
    horizontal_stop_right::Int
    horizontal_stop_left::Int
    vertical_stop_up::Int
    vertical_stop_down::Int
    mean_speed::Float64
    speed_variability::Float64
    max_sight_distance::Int
    planning_distance::Float64
    dt::Float64
    regen_eff::Float64

    # Traffic light timing
    green_duration_s::Float64
    amber_duration_s::Float64
    red_duration_s::Float64

    # Emission parameters
    brake_dispersion_radius::Int
    tyre_dispersion_radius::Int

    # Bus parameters
    bus_base_weight::Float64
    bus_occupancy::Float64

    # Traffic light entities
    vertical_light_entity::Ark.Entity
    horizontal_light_entity::Ark.Entity

    # Cached light states (updated each step)
    _h_light_state::UInt8
    _v_light_state::UInt8

    # RNG and cached data
    rng::Random.MersenneTwister
    scaled_pos::NamedTuple

    # Pre-allocated buffers
    _old_positions_buffer::Vector{Tuple{Int,Int}}

    # Entity data caches (indexed by _id, populated at start of each step)
    _cache_dir::Vector{UInt8}
    _cache_vtype::Vector{UInt8}
    _cache_fx::Vector{Float64}
    _cache_fy::Vector{Float64}
    _cache_gx::Vector{Int32}
    _cache_gy::Vector{Int32}
    _cache_speed::Vector{Float32}
    _cache_orig_speed::Vector{Float32}
    _cache_ts::Vector{UInt8}
    _cache_vlen::Vector{Int8}
    _cache_weight::Vector{Float32}

    # Conflict detection grid (pre-allocated, dims-sized)
    _intended_occ::Matrix{UInt32}

    # Conflict resolution buffers
    _conflict_eids::Vector{UInt32}
    _conflict_prio::Vector{Float64}
end

# Zero-alloc direction offset for position iteration
@inline function pos_offset(dir::UInt8)
    dir == DIR_RIGHT ? (-1, 0) :
    dir == DIR_LEFT  ? (1, 0) :
    dir == DIR_UP    ? (0, -1) : (0, 1)
end

# ============================================================================
# SECTION 4: PURE HELPER FUNCTIONS
# ============================================================================

function get_scaled_positions_ecs(dims::Tuple{Int,Int})
    width, height = dims
    center_x, center_y = width ÷ 2, height ÷ 2
    scale_x = width / 400.0
    scale_y = height / 400.0
    return (
        intersection_center = (center_x, center_y),
        left_lane_up = round(Int, 193 * scale_x),
        right_lane_up = round(Int, 197 * scale_x),
        left_lane_down = round(Int, 207 * scale_x),
        right_lane_down = round(Int, 203 * scale_x),
        left_lane_left = round(Int, 193 * scale_y),
        right_lane_left = round(Int, 197 * scale_y),
        left_lane_right = round(Int, 207 * scale_y),
        right_lane_right = round(Int, 203 * scale_y),
        bottom_wall = round(Int, (190 - 2) * scale_y),
        top_wall = round(Int, (210 + 2) * scale_y),
        left_wall = round(Int, (190 - 2) * scale_x),
        right_wall = round(Int, (210 + 2) * scale_x)
    )
end

function get_scaled_observer_position_ecs(dims::Tuple{Int,Int})
    width, height = dims
    return (round(Int, width * 0.48), round(Int, height * 0.45))
end

function get_light_durations_ecs(dt::Float64, green_s::Float64, amber_s::Float64, red_s::Float64;
                                 clearance_s::Float64=DEFAULT_CLEARANCE_S)
    green_steps = ceil(Int, green_s / dt)
    amber_steps = ceil(Int, amber_s / dt)
    clearance_steps = ceil(Int, clearance_s / dt)
    half_cycle = green_steps + amber_steps + clearance_steps
    cycle_steps = 2 * half_cycle
    return (green=green_steps, amber=amber_steps, clearance=clearance_steps,
            half_cycle=half_cycle, cycle=cycle_steps)
end

function calculate_adaptive_dt_ecs(mean_speed::Float64; speed_variability::Float64=0.1, target_distance::Float64=1.0)
    max_expected_speed = mean_speed + 2.0 * speed_variability
    dt = target_distance / max(max_expected_speed, 1.0)
    return clamp(dt, 0.05, 1.0)
end

@inline function get_vehicle_length(vtype::UInt8)::Int
    vtype == VT_SMART_CAR ? 2 :
    vtype == VT_CAR || vtype == VT_E_CAR ? 3 :
    vtype == VT_SUV || vtype == VT_E_SUV || vtype == VT_OBSERVER ? 4 : 7
end

@inline function get_post_turn_direction(dir::UInt8)::UInt8
    dir == DIR_UP ? DIR_LEFT :
    dir == DIR_DOWN ? DIR_RIGHT :
    dir == DIR_LEFT ? DIR_DOWN : DIR_UP
end

function calculate_occupied_positions_ecs(vtype::UInt8, direction::UInt8, start::Tuple{Int,Int})
    vlen = get_vehicle_length(vtype)
    sx, sy = start
    positions = Vector{Tuple{Int,Int}}(undef, vlen)
    @inbounds if direction == DIR_RIGHT
        for i in 0:vlen-1; positions[i+1] = (sx - i, sy); end
    elseif direction == DIR_LEFT
        for i in 0:vlen-1; positions[i+1] = (sx + i, sy); end
    elseif direction == DIR_UP
        for i in 0:vlen-1; positions[i+1] = (sx, sy - i); end
    else  # DIR_DOWN
        for i in 0:vlen-1; positions[i+1] = (sx, sy + i); end
    end
    return positions
end

@inline function get_ahead_from_front(x::Int, y::Int, dir::UInt8, dist::Int)
    dir == DIR_RIGHT ? (x + dist, y) :
    dir == DIR_LEFT  ? (x - dist, y) :
    dir == DIR_UP    ? (x, y + dist) : (x, y - dist)
end

@inline function get_lane_ecs(dir::UInt8, pos_x, pos_y)
    (dir == DIR_RIGHT || dir == DIR_LEFT) ? Int(pos_y) : Int(pos_x)
end

function get_other_lane_ecs(dir::UInt8, current_lane::Int, sp)
    if dir == DIR_RIGHT
        current_lane == sp.right_lane_right ? sp.left_lane_right : sp.right_lane_right
    elseif dir == DIR_LEFT
        current_lane == sp.left_lane_left ? sp.right_lane_left : sp.left_lane_left
    elseif dir == DIR_UP
        current_lane == sp.left_lane_up ? sp.right_lane_up : sp.left_lane_up
    else  # DIR_DOWN
        current_lane == sp.right_lane_down ? sp.left_lane_down : sp.right_lane_down
    end
end

@inline function get_position_in_lane_ecs(dir::UInt8, pos_x::Int, pos_y::Int, lane::Int)
    (dir == DIR_RIGHT || dir == DIR_LEFT) ? (pos_x, lane) : (lane, pos_y)
end

@inline function get_ahead_in_lane_ecs(dir::UInt8, pos_x::Int, pos_y::Int, lane::Int, dist::Int)
    dir == DIR_RIGHT ? (pos_x + dist, lane) :
    dir == DIR_LEFT  ? (pos_x - dist, lane) :
    dir == DIR_UP    ? (lane, pos_y + dist) : (lane, pos_y - dist)
end

@inline function is_occupied_ecs(x::Int, y::Int, eid::UInt32, res::SimState)
    w, h = res.dims
    wx = mod1(x, w); wy = mod1(y, h)
    @inbounds occ = res.occupancy_grid[wx, wy]
    return occ != UInt32(0) && occ != eid
end

function find_vehicle_at_ecs(x::Int, y::Int, eid::UInt32, world, res::SimState)
    w, h = res.dims
    wx = mod1(x, w); wy = mod1(y, h)
    @inbounds occ = res.occupancy_grid[wx, wy]
    if occ == UInt32(0) || occ == eid
        return nothing
    end
    entity = res.id_to_entity[occ]
    fp, vid, ts = get_components(world, entity, (FracPos, VehicleId, TurnState))
    vlen = get_vehicle_length(vid.vehicle_type)
    return (fx=fp.x, fy=fp.y, direction=vid.direction, vehicle_type=vid.vehicle_type,
            vehicle_length=vlen, turning_state=ts.turning_state, entity_id=occ)
end

function get_vehicle_rear_pos(dir::UInt8, head_x::Int, head_y::Int, vlen::Int)
    dir == DIR_RIGHT ? (head_x - vlen + 1, head_y) :
    dir == DIR_LEFT  ? (head_x + vlen - 1, head_y) :
    dir == DIR_UP    ? (head_x, head_y - vlen + 1) : (head_x, head_y + vlen - 1)
end

function find_rear_position_ecs(x::Int, y::Int, eid::UInt32, world, res::SimState)
    info = find_vehicle_at_ecs(x, y, eid, world, res)
    if info !== nothing
        w, h = res.dims
        # Use grid position from occupancy (head is stored in GridPos)
        entity = res.id_to_entity[info.entity_id]
        gp, = get_components(world, entity, (GridPos,))
        rear = get_vehicle_rear_pos(info.direction, Int(gp.x), Int(gp.y), info.vehicle_length)
        return (mod1(rear[1], w), mod1(rear[2], h))
    end
    return (x, y)
end

function is_in_intersection_ecs(x::Int, y::Int, dir::UInt8, res::SimState)
    if dir == DIR_RIGHT || dir == DIR_LEFT
        return x >= res.horizontal_stop_right && x <= res.horizontal_stop_left
    else
        return y >= res.vertical_stop_up && y <= res.vertical_stop_down
    end
end

function is_exit_blocked_ecs(dir::UInt8, pos_x::Int, pos_y::Int, vtype::UInt8,
                              in_intersection::Bool, eid::UInt32, res::SimState)
    if in_intersection; return false; end
    vlen = get_vehicle_length(vtype)
    w, h = res.dims
    if dir == DIR_RIGHT
        for cx in res.horizontal_stop_right:(res.horizontal_stop_left + vlen + 3)
            if is_occupied_ecs(cx, pos_y, eid, res); return true; end
        end
    elseif dir == DIR_LEFT
        for cx in (res.horizontal_stop_right - vlen - 3):res.horizontal_stop_left
            if is_occupied_ecs(cx, pos_y, eid, res); return true; end
        end
    elseif dir == DIR_UP
        for cy in res.vertical_stop_up:(res.vertical_stop_down + vlen + 3)
            if is_occupied_ecs(pos_x, cy, eid, res); return true; end
        end
    else  # DIR_DOWN
        for cy in (res.vertical_stop_up - vlen - 3):res.vertical_stop_down
            if is_occupied_ecs(pos_x, cy, eid, res); return true; end
        end
    end
    return false
end

function should_stop_at_position_ecs(dir::UInt8, in_intersection::Bool,
                                      ahead_x::Int, ahead_y::Int,
                                      pos_x::Int, pos_y::Int,
                                      vtype::UInt8, eid::UInt32, res::SimState)
    if in_intersection; return false; end

    sp = res.scaled_pos
    light_state = (dir == DIR_LEFT || dir == DIR_RIGHT) ? res._h_light_state : res._v_light_state
    at_stop_line = false

    if dir == DIR_RIGHT || dir == DIR_LEFT
        stopping_line = dir == DIR_RIGHT ? res.horizontal_stop_right : res.horizontal_stop_left
        rl1 = dir == DIR_RIGHT ? sp.right_lane_right : sp.left_lane_left
        rl2 = dir == DIR_RIGHT ? sp.left_lane_right : sp.right_lane_left
        if ahead_y == rl1 || ahead_y == rl2
            at_stop_line = (dir == DIR_RIGHT && ahead_x >= stopping_line && ahead_x < stopping_line + 1) ||
                           (dir == DIR_LEFT && ahead_x <= stopping_line && ahead_x > stopping_line - 1)
        end
    else
        stopping_line = dir == DIR_UP ? res.vertical_stop_up : res.vertical_stop_down
        rl1 = dir == DIR_UP ? sp.left_lane_up : sp.right_lane_down
        rl2 = dir == DIR_UP ? sp.right_lane_up : sp.left_lane_down
        if ahead_x == rl1 || ahead_x == rl2
            at_stop_line = (dir == DIR_UP && ahead_y >= stopping_line && ahead_y < stopping_line + 1) ||
                           (dir == DIR_DOWN && ahead_y <= stopping_line && ahead_y > stopping_line - 1)
        end
    end

    if at_stop_line
        light_says_stop = (light_state == TL_RED || light_state == TL_AMBER)
        exit_blocked = is_exit_blocked_ecs(dir, pos_x, pos_y, vtype, in_intersection, eid, res)
        return light_says_stop || exit_blocked
    end
    return false
end

function should_turn_left_ecs(dir::UInt8, ts_state::UInt8, fx::Float64, fy::Float64,
                               pos_x::Int, pos_y::Int, turn_intention::UInt8, res::SimState)
    if ts_state != TS_NOT_TURNING; return false, turn_intention; end

    distance_to_intersection = if dir == DIR_UP
        Float64(res.vertical_stop_up) - fy
    elseif dir == DIR_DOWN
        fy - Float64(res.vertical_stop_down)
    elseif dir == DIR_LEFT
        fx - Float64(res.horizontal_stop_left)
    else
        Float64(res.horizontal_stop_right) - fx
    end

    sp = res.scaled_pos
    current_lane = get_lane_ecs(dir, pos_x, pos_y)
    is_in_left_lane = (dir == DIR_UP && current_lane == sp.left_lane_up) ||
                      (dir == DIR_DOWN && current_lane == sp.left_lane_down) ||
                      (dir == DIR_LEFT && current_lane == sp.left_lane_left) ||
                      (dir == DIR_RIGHT && current_lane == sp.left_lane_right)

    at_intersection = if dir == DIR_UP || dir == DIR_DOWN
        fy >= res.vertical_stop_up && fy <= res.vertical_stop_down
    else
        fx >= res.horizontal_stop_right && fx <= res.horizontal_stop_left
    end

    if is_in_left_lane && at_intersection && distance_to_intersection < 0
        light_state = (dir == DIR_UP || dir == DIR_DOWN) ? res._v_light_state : res._h_light_state
        if light_state != TL_GREEN; return false, turn_intention; end

        if distance_to_intersection > -2.0
            if dir == DIR_UP
                turn_intention = TI_LEFT
            else
                turn_intention = rand(res.rng) < 0.2 ? TI_LEFT : TI_STRAIGHT
            end
        end
        return turn_intention == TI_LEFT, turn_intention
    end
    return false, turn_intention
end

function should_move_to_left_lane_ecs(dir::UInt8, ti::UInt8, fx::Float64, fy::Float64,
                                       pos_x::Int, pos_y::Int, plan_dist::Int, res::SimState)
    if ti != TI_LEFT; return false; end

    dist_to_inter = if dir == DIR_UP
        Float64(res.vertical_stop_up) - fy
    elseif dir == DIR_DOWN
        fy - Float64(res.vertical_stop_down)
    elseif dir == DIR_LEFT
        fx - Float64(res.horizontal_stop_left)
    else
        Float64(res.horizontal_stop_right) - fx
    end

    sp = res.scaled_pos
    current_lane = get_lane_ecs(dir, pos_x, pos_y)
    is_in_left_lane = (dir == DIR_UP && current_lane == sp.left_lane_up) ||
                      (dir == DIR_DOWN && current_lane == sp.left_lane_down) ||
                      (dir == DIR_LEFT && current_lane == sp.left_lane_left) ||
                      (dir == DIR_RIGHT && current_lane == sp.left_lane_right)

    return dist_to_inter <= plan_dist && !is_in_left_lane
end

function would_create_body_overlap_ecs(vtype::UInt8, dir::UInt8, target_pos,
                                        eid::UInt32, res::SimState, world;
                                        target_direction::Union{UInt8,Nothing}=nothing)
    w, h = res.dims
    effective_dir = target_direction !== nothing ? target_direction : dir
    vlen = get_vehicle_length(vtype)
    dx, dy = pos_offset(effective_dir)
    hx, hy = Int(target_pos[1]), Int(target_pos[2])

    for k in 0:vlen-1
        cx = hx + k * dx; cy = hy + k * dy
        wx = mod1(cx, w); wy = mod1(cy, h)
        @inbounds occupant = res.occupancy_grid[wx, wy]
        if occupant != UInt32(0) && occupant != eid
            @inbounds other_dir = res._cache_dir[occupant]
            if other_dir == 0x00  # cache not yet populated
                entity = res.id_to_entity[occupant]
                other_vid, = get_components(world, entity, (VehicleId,))
                other_dir = other_vid.direction
            end
            is_perp = (dir == DIR_LEFT || dir == DIR_RIGHT) ?
                      (other_dir == DIR_UP || other_dir == DIR_DOWN) :
                      (other_dir == DIR_LEFT || other_dir == DIR_RIGHT)
            if is_perp; return true; end
        end
    end
    return false
end

function generate_speed_ecs(res::SimState)
    ms = res.mean_speed
    sv = res.speed_variability
    μ = log(ms^2 / sqrt(ms^2 + sv^2))
    σ = sqrt(log(1 + (sv^2 / ms^2)))
    return Float32(rand(res.rng, LogNormal(μ, σ)))
end

# ============================================================================
# SECTION 5: EMISSION FUNCTIONS
# ============================================================================

function disperse_in_cone_ecs!(grid::Matrix{Float64}, origin, dir::UInt8,
                                amount::Float64, dims::Tuple{Int,Int},
                                decay_table::Matrix{Float64}, radius::Int)
    ox, oy = origin[1], origin[2]
    width, height = dims
    @inbounds for xo in -radius:radius
        for yo in -radius:radius
            if xo == 0 && yo == 0; continue; end
            is_in_cone = if dir == DIR_RIGHT
                xo <= 0 && abs(yo) <= abs(xo)
            elseif dir == DIR_LEFT
                xo >= 0 && abs(yo) <= abs(xo)
            elseif dir == DIR_DOWN
                yo >= 0 && abs(xo) <= abs(yo)
            else  # DIR_UP
                yo <= 0 && abs(xo) <= abs(yo)
            end
            if is_in_cone
                nx = ox + xo; ny = oy + yo
                if 1 <= nx <= width && 1 <= ny <= height
                    grid[nx, ny] += amount * decay_table[xo + radius + 1, yo + radius + 1]
                end
            end
        end
    end
end

function generate_brake_emissions_ecs!(res::SimState, vtype::UInt8, dir::UInt8,
                                        pos_x::Int, pos_y::Int, weight::Float32,
                                        prev_speed::Float32, cur_speed::Float32)
    speed_reduction = prev_speed - cur_speed
    if speed_reduction <= 0 || prev_speed == 0; return; end

    vlen = get_vehicle_length(vtype)
    dx, dy = pos_offset(dir)
    w, h = res.dims
    r = res.brake_dispersion_radius
    decay_table = GAUSSIAN_DECAY_TABLE_ECS[r]

    weight_factor = Float64(weight) / 1500.0
    brake_factor = Float64(speed_reduction) / Float64(prev_speed)
    emission_factor = weight_factor * brake_factor
    if vtype == VT_E_CAR || vtype == VT_E_SUV || vtype == VT_OBSERVER
        emission_factor *= (1.0 - res.regen_eff)
    end

    for k in 0:vlen-1
        px = pos_x + k * dx; py = pos_y + k * dy
        wpx = mod1(px, w); wpy = mod1(py, h)
        disperse_in_cone_ecs!(res.brake_emissions, (wpx, wpy), dir, emission_factor, res.dims, decay_table, r)
    end
end

function generate_tyre_emissions_ecs!(res::SimState, vtype::UInt8, dir::UInt8,
                                       pos_x::Int, pos_y::Int, weight::Float32,
                                       cur_speed::Float32, prev_speed::Float32)
    vlen = get_vehicle_length(vtype)
    dx, dy = pos_offset(dir)
    w, h = res.dims
    r = res.tyre_dispersion_radius
    decay_table = GAUSSIAN_DECAY_TABLE_ECS[r]

    weight_factor = Float64(weight) / 1500.0
    speed_factor = Float64(cur_speed) / 10.0
    accel_factor = abs(Float64(cur_speed) - Float64(prev_speed)) / 2.0
    emission_factor = (speed_factor + accel_factor) * weight_factor

    for k in 0:vlen-1
        px = pos_x + k * dx; py = pos_y + k * dy
        wpx = mod1(px, w); wpy = mod1(py, h)
        disperse_in_cone_ecs!(res.tyre_emissions, (wpx, wpy), dir, emission_factor, res.dims, decay_table, r)
    end
end

# ============================================================================
# SECTION 6: TRAFFIC LIGHT SYSTEM
# ============================================================================

function system_traffic_lights!(world, res::SimState)
    durations = get_light_durations_ecs(res.dt, res.green_duration_s, res.amber_duration_s, res.red_duration_s)
    for (entities, tl) in Query(world, (TrafficLightComp,))
        for i in eachindex(entities)
            tc = tl[i]
            new_counter = tc.time_counter + Int32(1)
            cp = new_counter % durations.cycle
            new_state = if tc.light_type == LT_HORIZONTAL
                cp < durations.green ? TL_GREEN :
                cp < durations.green + durations.amber ? TL_AMBER : TL_RED
            else
                cp < durations.half_cycle ? TL_RED :
                cp < durations.half_cycle + durations.green ? TL_GREEN :
                cp < durations.half_cycle + durations.green + durations.amber ? TL_AMBER : TL_RED
            end
            tl[i] = TrafficLightComp(new_state, tc.light_type, new_counter)
        end
    end
    # Cache light states
    h_tlc, = get_components(world, res.horizontal_light_entity, (TrafficLightComp,))
    v_tlc, = get_components(world, res.vertical_light_entity, (TrafficLightComp,))
    res._h_light_state = h_tlc.state
    res._v_light_state = v_tlc.state
end

# ============================================================================
# SECTION 7: PLAN MOVES SYSTEM
# ============================================================================

function plan_turn_move_logic!(cur_speed::Float32, ts_state::UInt8, tp::Float32,
                                dir::UInt8, pos_x::Int, pos_y::Int,
                                fx::Float64, fy::Float64,
                                vtype::UInt8, eid::UInt32, world, res::SimState)
    if ts_state == TS_NOT_TURNING
        ts_state = TS_TURNING
        tp = Float32(0.0)
        cur_speed = Float32(Float64(cur_speed) * 0.6)
    end

    new_dir = get_post_turn_direction(dir)
    vlen = get_vehicle_length(vtype)
    sp = res.scaled_pos

    jump_pos = if dir == DIR_UP  # turning to LEFT
        (pos_x - vlen, sp.left_lane_left)
    elseif dir == DIR_DOWN  # turning to RIGHT
        (pos_x + vlen, sp.left_lane_right)
    elseif dir == DIR_LEFT  # turning to DOWN
        (sp.left_lane_down, pos_y - vlen)
    else  # DIR_RIGHT, turning to UP
        (sp.left_lane_up, pos_y + vlen)
    end

    if would_create_body_overlap_ecs(vtype, dir, jump_pos, eid, res, world; target_direction=new_dir)
        cur_speed = Float32(0.0)
        res.intended_moves[eid] = (Int32(pos_x), Int32(pos_y))
        res.intended_valid[eid] = true
        res.intended_vehicle_type[eid] = vtype
        res.intended_direction[eid] = new_dir
    else
        res.intended_moves[eid] = (Int32(jump_pos[1]), Int32(jump_pos[2]))
        res.intended_valid[eid] = true
        res.intended_vehicle_type[eid] = vtype
        res.intended_direction[eid] = new_dir
        fx = Float64(jump_pos[1])
        fy = Float64(jump_pos[2])
        tp = Float32(1.0)
    end

    return cur_speed, ts_state, tp, fx, fy
end

function compute_actual_distance(dir::UInt8, fx::Float64, fy::Float64,
                                  is_vehicle::Bool, is_stop::Bool,
                                  ahead_info, w::Int, h::Int, res::SimState)
    if is_vehicle && ahead_info !== nothing
        ahead_fx = ahead_info.fx
        ahead_fy = ahead_info.fy
        ahead_len = ahead_info.vehicle_length

        frac_rear = if ahead_info.direction == DIR_RIGHT
            ahead_fx - ahead_len + 1
        elseif ahead_info.direction == DIR_LEFT
            ahead_fx + ahead_len - 1
        elseif ahead_info.direction == DIR_UP
            ahead_fy - ahead_len + 1
        else
            ahead_fy + ahead_len - 1
        end

        raw_dist = if dir == DIR_RIGHT
            frac_rear - fx
        elseif dir == DIR_LEFT
            fx - frac_rear
        elseif dir == DIR_UP
            frac_rear - fy
        else
            fy - frac_rear
        end

        wrap_dim = Float64((dir == DIR_UP || dir == DIR_DOWN) ? h : w)
        if raw_dist < -(wrap_dim / 2); raw_dist += wrap_dim
        elseif raw_dist > (wrap_dim / 2); raw_dist -= wrap_dim; end
        return raw_dist

    elseif is_vehicle
        # Fallback grid-based
        rear_pos = find_rear_position_ecs(0, 0, eid_placeholder, nothing, res)  # won't be called
        raw_dist = if dir == DIR_RIGHT; Float64(rear_pos[1]) - fx
        elseif dir == DIR_LEFT; fx - Float64(rear_pos[1])
        elseif dir == DIR_UP; Float64(rear_pos[2]) - fy
        else; fy - Float64(rear_pos[2]); end
        wrap_dim = Float64((dir == DIR_UP || dir == DIR_DOWN) ? h : w)
        if raw_dist < -(wrap_dim / 2); raw_dist += wrap_dim
        elseif raw_dist > (wrap_dim / 2); raw_dist -= wrap_dim; end
        return raw_dist

    else  # Stop line
        raw_dist = if dir == DIR_RIGHT; Float64(res.horizontal_stop_right) - fx
        elseif dir == DIR_LEFT; fx - Float64(res.horizontal_stop_left)
        elseif dir == DIR_UP; Float64(res.vertical_stop_up) - fy
        else; fy - Float64(res.vertical_stop_down); end
        wrap_dim = Float64((dir == DIR_UP || dir == DIR_DOWN) ? h : w)
        if raw_dist < -(wrap_dim / 2); raw_dist += wrap_dim
        elseif raw_dist > (wrap_dim / 2); raw_dist -= wrap_dim; end
        return raw_dist
    end
end

# Per-entity planning logic — extracted as function barrier for type inference
function _plan_vehicle!(
    eid::UInt32, pos_x::Int, pos_y::Int,
    fx::Float64, fy::Float64,
    direction::UInt8, vtype::UInt8, vweight::Float32, plan_dist::Int,
    cur_speed::Float32, orig_speed::Float32,
    ts_state::UInt8, ti::UInt8, tp::Float32, in_inter::Bool,
    world, res::SimState, w::Int, h::Int, wf::Float64, hf::Float64,
    vm_col, fp_col, ts_col, idx::Int
)
    prev_speed = cur_speed
    did_plan = false

    # === TURNING VEHICLES ===
    if ts_state == TS_TURNING
        cur_speed, ts_state, tp, fx, fy = plan_turn_move_logic!(
            cur_speed, ts_state, tp, direction, pos_x, pos_y, fx, fy,
            vtype, eid, world, res)
        cur_speed = cur_speed::Float32; fx = fx::Float64; fy = fy::Float64
        tp = tp::Float32; ts_state = ts_state::UInt8
        did_plan = true
    end

    if !did_plan
        # === GRADUAL BRAKING FOR LEFT TURN ===
        if ti == TI_LEFT && ts_state == TS_NOT_TURNING
            dist_to_inter = if direction == DIR_UP
                Float64(res.vertical_stop_up) - fy
            elseif direction == DIR_DOWN
                fy - Float64(res.vertical_stop_down)
            elseif direction == DIR_LEFT
                fx - Float64(res.horizontal_stop_left)
            else
                Float64(res.horizontal_stop_right) - fx
            end
            braking_distance = 10.0
            if dist_to_inter > 0 && dist_to_inter < braking_distance
                brake_factor = 0.6 + (0.4 * (dist_to_inter / braking_distance))
                cur_speed = Float32(Float64(orig_speed) * brake_factor)
            end
        end

        # === CHECK IF SHOULD TURN LEFT ===
        should_turn, ti = should_turn_left_ecs(direction, ts_state, fx, fy, pos_x, pos_y, ti, res)
        ti = ti::UInt8
        if should_turn
            cur_speed, ts_state, tp, fx, fy = plan_turn_move_logic!(
                cur_speed, ts_state, tp, direction, pos_x, pos_y, fx, fy,
                vtype, eid, world, res)
            cur_speed = cur_speed::Float32; fx = fx::Float64; fy = fy::Float64
            tp = tp::Float32; ts_state = ts_state::UInt8
            did_plan = true
        end
    end

    if !did_plan
        # === SCAN AHEAD FOR OBSTACLES ===
        cur_speed = cur_speed::Float32  # reassert type for inference stability
        max_check = max(Int(res.max_sight_distance), ceil(Int, Float64(cur_speed)) + 1)

        for dist in 1:max_check
            ahead_x, ahead_y = get_ahead_from_front(pos_x, pos_y, direction, dist)

            is_stop = should_stop_at_position_ecs(direction, in_inter, ahead_x, ahead_y,
                                                  pos_x, pos_y, vtype, eid, res)

            # Check for vehicle at ahead position
            ahead_wx = mod1(ahead_x, w); ahead_wy = mod1(ahead_y, h)
            @inbounds ahead_occ = res.occupancy_grid[ahead_wx, ahead_wy]
            raw_is_vehicle = ahead_occ != UInt32(0) && ahead_occ != eid

            # Find rear of vehicle ahead (O(1) cache lookup)
            is_vehicle = false
            if raw_is_vehicle
                @inbounds other_vlen = Int(res._cache_vlen[ahead_occ])
                @inbounds other_dir_c = res._cache_dir[ahead_occ]
                @inbounds rear = get_vehicle_rear_pos(other_dir_c, Int(res._cache_gx[ahead_occ]), Int(res._cache_gy[ahead_occ]), other_vlen)
                rear_wx = mod1(rear[1], w); rear_wy = mod1(rear[2], h)
                @inbounds rear_occ = res.occupancy_grid[rear_wx, rear_wy]
                is_vehicle = rear_occ != UInt32(0) && rear_occ != eid
            end

            if is_vehicle || is_stop
                # Read ahead vehicle data (flat locals avoid Union boxing)
                _av_fx = 0.0; _av_fy = 0.0; _av_dir = UInt8(0)
                _av_vlen = 0; _av_turning = UInt8(0)
                if raw_is_vehicle
                    _aent = res.id_to_entity[ahead_occ]
                    _afp, _ats = get_components(world, _aent, (FracPos, TurnState))
                    _av_fx = _afp.x; _av_fy = _afp.y
                    @inbounds _av_dir = res._cache_dir[ahead_occ]
                    @inbounds _av_vlen = Int(res._cache_vlen[ahead_occ])
                    _av_turning = _ats.turning_state
                end

                actual_distance = if is_vehicle && raw_is_vehicle
                    # Fractional distance to rear of vehicle ahead
                    frac_rear = if _av_dir == DIR_RIGHT
                        _av_fx - _av_vlen + 1
                    elseif _av_dir == DIR_LEFT
                        _av_fx + _av_vlen - 1
                    elseif _av_dir == DIR_UP
                        _av_fy - _av_vlen + 1
                    else
                        _av_fy + _av_vlen - 1
                    end

                    raw_d = if direction == DIR_RIGHT; frac_rear - fx
                    elseif direction == DIR_LEFT; fx - frac_rear
                    elseif direction == DIR_UP; frac_rear - fy
                    else; fy - frac_rear; end

                    wrap_dim = Float64((direction == DIR_UP || direction == DIR_DOWN) ? h : w)
                    if raw_d < -(wrap_dim / 2); raw_d += wrap_dim
                    elseif raw_d > (wrap_dim / 2); raw_d -= wrap_dim; end
                    raw_d

                elseif is_vehicle
                    # Grid-based fallback
                    rear_pos = find_rear_position_ecs(ahead_x, ahead_y, eid, world, res)
                    raw_d = if direction == DIR_RIGHT; Float64(rear_pos[1]) - fx
                    elseif direction == DIR_LEFT; fx - Float64(rear_pos[1])
                    elseif direction == DIR_UP; Float64(rear_pos[2]) - fy
                    else; fy - Float64(rear_pos[2]); end
                    wrap_dim = Float64((direction == DIR_UP || direction == DIR_DOWN) ? h : w)
                    if raw_d < -(wrap_dim / 2); raw_d += wrap_dim
                    elseif raw_d > (wrap_dim / 2); raw_d -= wrap_dim; end
                    raw_d

                else
                    # Stop line
                    raw_d = if direction == DIR_RIGHT; Float64(res.horizontal_stop_right) - fx
                    elseif direction == DIR_LEFT; fx - Float64(res.horizontal_stop_left)
                    elseif direction == DIR_UP; Float64(res.vertical_stop_up) - fy
                    else; fy - Float64(res.vertical_stop_down); end
                    wrap_dim = Float64((direction == DIR_UP || direction == DIR_DOWN) ? h : w)
                    if raw_d < -(wrap_dim / 2); raw_d += wrap_dim
                    elseif raw_d > (wrap_dim / 2); raw_d -= wrap_dim; end
                    raw_d
                end

                is_turning_ahead = raw_is_vehicle && _av_turning == TS_TURNING
                min_distance = 1.0
                stop_buffer = 0.0
                dist_will_move = Float64(cur_speed) * res.dt

                if actual_distance <= min_distance + stop_buffer
                    cur_speed = Float32(0.0)
                elseif actual_distance <= min_distance + stop_buffer + dist_will_move
                    cur_speed = Float32(0.0)
                else
                    slowdown = (actual_distance - min_distance) / max_check
                    reaction = Float64(rand(res.rng, REACTION_DIST_ECS))
                    if is_turning_ahead
                        cur_speed = Float32(Float64(orig_speed) * slowdown * reaction * 0.5)
                    else
                        cur_speed = Float32(Float64(orig_speed) * slowdown * reaction)
                    end
                end
                break
            else
                if cur_speed < orig_speed
                    cur_speed = min(orig_speed, cur_speed + MAX_ACCELERATION_ECS)
                end
            end
        end

        # === LANE CHANGE ===
        if !did_plan && should_move_to_left_lane_ecs(direction, ti, fx, fy, pos_x, pos_y, plan_dist, res)
            sp = res.scaled_pos
            left_lane = if direction == DIR_UP; sp.left_lane_up
            elseif direction == DIR_DOWN; sp.left_lane_down
            elseif direction == DIR_LEFT; sp.left_lane_left
            else; sp.left_lane_right; end

            ll_pos = get_position_in_lane_ecs(direction, pos_x, pos_y, left_lane)
            if !is_occupied_ecs(ll_pos[1], ll_pos[2], eid, res)
                res.intended_moves[eid] = (Int32(ll_pos[1]), Int32(ll_pos[2]))
                res.intended_valid[eid] = true
                res.intended_vehicle_type[eid] = vtype
                eff_dir = (ts_state == TS_TURNING) ? get_post_turn_direction(direction) : direction
                res.intended_direction[eid] = eff_dir
                did_plan = true
            end
        end
    end

    if !did_plan
        # === GENERATE EMISSIONS ===
        cur_speed = cur_speed::Float32; fx = fx::Float64; fy = fy::Float64
        if cur_speed < prev_speed
            generate_brake_emissions_ecs!(res, vtype, direction, pos_x, pos_y, vweight, prev_speed, cur_speed)
        end
        generate_tyre_emissions_ecs!(res, vtype, direction, pos_x, pos_y, vweight, cur_speed, prev_speed)

        # === FRACTIONAL MOVEMENT ===
        if fx > wf; fx = 1.0 + mod(fx - 1.0, wf); end
        if fx < 1.0; fx = 1.0 + mod(fx - 1.0, wf); end
        if fy > hf; fy = 1.0 + mod(fy - 1.0, hf); end
        if fy < 1.0; fy = 1.0 + mod(fy - 1.0, hf); end

        dist_to_move = Float64(cur_speed) * res.dt

        if Float64(cur_speed) < 0.01
            gx = Int32(mod1(round(Int, fx), w))
            gy = Int32(mod1(round(Int, fy), h))
            res.intended_moves[eid] = (gx, gy)
            res.intended_valid[eid] = true
            res.intended_vehicle_type[eid] = vtype
            eff_dir = (ts_state == TS_TURNING) ? get_post_turn_direction(direction) : direction
            res.intended_direction[eid] = eff_dir
        else
            new_fx, new_fy = if direction == DIR_RIGHT; (fx + dist_to_move, fy)
            elseif direction == DIR_LEFT; (fx - dist_to_move, fy)
            elseif direction == DIR_UP; (fx, fy + dist_to_move)
            else; (fx, fy - dist_to_move); end

            if new_fx > wf; new_fx = 1.0 + mod(new_fx - 1.0, wf); end
            if new_fx < 1.0; new_fx = 1.0 + mod(new_fx - 1.0, wf); end
            if new_fy > hf; new_fy = 1.0 + mod(new_fy - 1.0, hf); end
            if new_fy < 1.0; new_fy = 1.0 + mod(new_fy - 1.0, hf); end

            gx = Int32(mod1(round(Int, new_fx), w))
            gy = Int32(mod1(round(Int, new_fy), h))

            if would_create_body_overlap_ecs(vtype, direction, (gx, gy), eid, res, world)
                cur_speed = Float32(0.0)
                gx = Int32(mod1(round(Int, fx), w))
                gy = Int32(mod1(round(Int, fy), h))
            else
                fx = new_fx; fy = new_fy
            end

            res.intended_moves[eid] = (gx, gy)
            res.intended_valid[eid] = true
            res.intended_vehicle_type[eid] = vtype
            eff_dir = (ts_state == TS_TURNING) ? get_post_turn_direction(direction) : direction
            res.intended_direction[eid] = eff_dir
        end
    end

    # Write back directly to component columns (avoids return tuple boxing)
    vm_col[idx] = VehicleMotion(cur_speed, orig_speed, prev_speed)
    fp_col[idx] = FracPos(fx, fy)
    ts_col[idx] = TurnState(ts_state, ti, tp, in_inter)
    return nothing
end

function system_plan_moves!(world, res::SimState)
    fill!(res.intended_valid, false)
    w, h = res.dims
    wf, hf = Float64(w), Float64(h)

    # Populate entity caches for O(1) lookups in obstacle scanning and conflict resolution
    for (entities, gp, vid) in Query(world, (GridPos, VehicleId))
        for i in eachindex(entities)
            eid = entities[i]._id
            @inbounds begin
                res._cache_dir[eid] = vid[i].direction
                res._cache_vtype[eid] = vid[i].vehicle_type
                res._cache_gx[eid] = gp[i].x
                res._cache_gy[eid] = gp[i].y
                res._cache_vlen[eid] = Int8(get_vehicle_length(vid[i].vehicle_type))
            end
        end
    end

    for (entities, gp, fp, vid, vm, ts) in Query(world, (GridPos, FracPos, VehicleId, VehicleMotion, TurnState))
        for i in eachindex(entities)
            eid = entities[i]._id
            g = gp[i]; f = fp[i]; v = vid[i]; m = vm[i]; t = ts[i]

            _plan_vehicle!(
                eid, Int(g.x), Int(g.y), f.x, f.y,
                v.direction, v.vehicle_type, v.weight, Int(v.planning_distance),
                m.speed, m.original_speed,
                t.turning_state, t.turn_intention, t.turn_progress, t.in_intersection,
                world, res, w, h, wf, hf,
                vm, fp, ts, i)
        end
    end
end

# ============================================================================
# SECTION 8: CONFLICT RESOLUTION SYSTEM
# ============================================================================

@inline function calculate_priority_ecs(eid::UInt32, contested_cell_x::Int, contested_cell_y::Int,
                                         world, res::SimState)
    entity = res.id_to_entity[eid]
    fp, vm, ts = get_components(world, entity, (FracPos, VehicleMotion, TurnState))

    # Use cache for stable fields (GridPos/VehicleId don't change in plan_moves)
    @inbounds gx = res._cache_gx[eid]
    @inbounds gy = res._cache_gy[eid]
    @inbounds dir = res._cache_dir[eid]

    priority = 0.0

    # Rule 1: In-intersection priority
    intended = res.intended_moves[eid]
    intended_in = is_in_intersection_ecs(Int(intended[1]), Int(intended[2]), dir, res)
    current_in = is_in_intersection_ecs(Int(gx), Int(gy), dir, res)
    if current_in || intended_in
        priority += 1000.0
    end

    # Rule 1.5: Turning vehicles
    if ts.turning_state == TS_TURNING
        priority += 500.0
    end

    # Rule 2: Distance to contested cell
    dist = sqrt((Float64(gx) - contested_cell_x)^2 + (Float64(gy) - contested_cell_y)^2)
    priority += 100.0 / (dist + 1.0)

    # Rule 3: Green light bonus
    light_state = (dir == DIR_LEFT || dir == DIR_RIGHT) ? res._h_light_state : res._v_light_state
    if light_state == TL_GREEN; priority += 50.0
    elseif light_state == TL_AMBER; priority += 25.0; end

    # Rule 4: Speed bonus
    if vm.speed > 0; priority += 10.0; end

    # Rule 5: Queue position
    queue_pos = if dir == DIR_UP; fp.y
    elseif dir == DIR_DOWN; -fp.y
    elseif dir == DIR_RIGHT; fp.x
    else; -fp.x; end
    priority += queue_pos * 2.0

    return priority
end

function is_position_safe_ecs(eid::UInt32, test_pos, res::SimState)
    vtype = res.intended_vehicle_type[eid]
    dir = res.intended_direction[eid]
    vlen = get_vehicle_length(vtype)
    tdx, tdy = pos_offset(dir)
    thx, thy = Int(test_pos[1]), Int(test_pos[2])

    @inbounds for other_eid in UInt32(1):UInt32(length(res.intended_valid))
        if !res.intended_valid[other_eid] || other_eid == eid; continue; end
        other_vtype = res.intended_vehicle_type[other_eid]
        other_dir = res.intended_direction[other_eid]
        other_target = res.intended_moves[other_eid]
        other_vlen = get_vehicle_length(other_vtype)
        odx, ody = pos_offset(other_dir)
        ohx, ohy = Int(other_target[1]), Int(other_target[2])
        for k in 0:vlen-1
            cx = thx + k * tdx; cy = thy + k * tdy
            for ok in 0:other_vlen-1
                ocx = ohx + ok * odx; ocy = ohy + ok * ody
                if cx == ocx && cy == ocy; return false; end
            end
        end
    end
    return true
end

function find_safe_fallback_ecs(eid::UInt32, world, res::SimState)
    @inbounds gx = res._cache_gx[eid]
    @inbounds gy = res._cache_gy[eid]
    @inbounds dir = res._cache_dir[eid]
    current_pos = (gx, gy)

    if is_position_safe_ecs(eid, current_pos, res)
        return current_pos
    end

    backward = if dir == DIR_RIGHT; (Int32(gx - 1), gy)
    elseif dir == DIR_LEFT; (Int32(gx + 1), gy)
    elseif dir == DIR_UP; (gx, Int32(gy - 1))
    else; (gx, Int32(gy + 1)); end

    if is_position_safe_ecs(eid, backward, res)
        return backward
    end
    return current_pos
end

function system_resolve_conflicts!(world, res::SimState)
    w, h = res.dims
    n_intended = length(res.intended_valid)
    max_iterations = 10

    for iteration in 0:max_iterations-1
        # ========== DETECT CONFLICTS via _intended_occ grid ==========
        fill!(res._intended_occ, UInt32(0))
        has_any_conflict = false

        # First pass: populate grid (first entity per cell wins)
        @inbounds for eid in UInt32(1):UInt32(n_intended)
            if !res.intended_valid[eid]; continue; end
            vtype = res.intended_vehicle_type[eid]
            dir = res.intended_direction[eid]
            target = res.intended_moves[eid]
            vlen = get_vehicle_length(vtype)
            dx, dy = pos_offset(dir)
            hx, hy = Int(target[1]), Int(target[2])
            for k in 0:vlen-1
                cx = hx + k * dx; cy = hy + k * dy
                wx = mod1(cx, w); wy = mod1(cy, h)
                existing = res._intended_occ[wx, wy]
                if existing == UInt32(0)
                    res._intended_occ[wx, wy] = eid
                elseif existing != eid
                    has_any_conflict = true
                end
            end
        end

        if !has_any_conflict; break; end

        # Second pass: find conflicted entities and compute max priority
        @inbounds for i in 1:n_intended; res._conflict_prio[i] = -Inf; end

        @inbounds for eid in UInt32(1):UInt32(n_intended)
            if !res.intended_valid[eid]; continue; end
            vtype = res.intended_vehicle_type[eid]
            dir = res.intended_direction[eid]
            target = res.intended_moves[eid]
            vlen = get_vehicle_length(vtype)
            dx, dy = pos_offset(dir)
            hx, hy = Int(target[1]), Int(target[2])
            for k in 0:vlen-1
                cx = hx + k * dx; cy = hy + k * dy
                wx = mod1(cx, w); wy = mod1(cy, h)
                occ = res._intended_occ[wx, wy]
                if occ != UInt32(0) && occ != eid
                    # eid conflicts with occ at cell (wx, wy)
                    p_eid = calculate_priority_ecs(eid, wx, wy, world, res)
                    res._conflict_prio[eid] = max(res._conflict_prio[eid], p_eid)
                    p_occ = calculate_priority_ecs(occ, wx, wy, world, res)
                    res._conflict_prio[occ] = max(res._conflict_prio[occ], p_occ)
                end
            end
        end

        # Collect conflicted entity ids into pre-allocated buffer
        n_conflicted = 0
        @inbounds for eid in UInt32(1):UInt32(n_intended)
            if res._conflict_prio[eid] > -Inf
                n_conflicted += 1
                res._conflict_eids[n_conflicted] = eid
            end
        end

        if iteration == max_iterations - 1
            # Last resort: revert all conflicted to current grid position
            for ci in 1:n_conflicted
                aid = res._conflict_eids[ci]
                @inbounds res.intended_moves[aid] = (res._cache_gx[aid], res._cache_gy[aid])
            end
            break
        end

        # ========== RESOLVE BY PRIORITY ==========
        # Sort conflicted entities by priority descending (insertion sort — n is small)
        for i in 2:n_conflicted
            @inbounds key_eid = res._conflict_eids[i]
            @inbounds key_prio = res._conflict_prio[key_eid]
            j = i - 1
            @inbounds while j >= 1 && res._conflict_prio[res._conflict_eids[j]] < key_prio
                res._conflict_eids[j+1] = res._conflict_eids[j]
                j -= 1
            end
            @inbounds res._conflict_eids[j+1] = key_eid
        end

        # Higher priority keeps position; lower priority gets fallback
        for ci in 1:n_conflicted
            @inbounds agent_id = res._conflict_eids[ci]
            vtype = res.intended_vehicle_type[agent_id]
            dir = res.intended_direction[agent_id]
            target = res.intended_moves[agent_id]
            vlen = get_vehicle_length(vtype)
            adx, ady = pos_offset(dir)
            ahx, ahy = Int(target[1]), Int(target[2])

            has_conflict = false
            for hi in 1:ci-1
                @inbounds other_id = res._conflict_eids[hi]
                other_vtype = res.intended_vehicle_type[other_id]
                other_dir = res.intended_direction[other_id]
                other_target = res.intended_moves[other_id]
                other_vlen = get_vehicle_length(other_vtype)
                odx, ody = pos_offset(other_dir)
                ohx, ohy = Int(other_target[1]), Int(other_target[2])
                for k in 0:vlen-1
                    cx = ahx + k * adx; cy = ahy + k * ady
                    for ok in 0:other_vlen-1
                        ocx = ohx + ok * odx; ocy = ohy + ok * ody
                        if cx == ocx && cy == ocy
                            has_conflict = true; @goto done_check
                        end
                    end
                end
            end
            @label done_check

            if has_conflict
                fallback = find_safe_fallback_ecs(agent_id, world, res)
                res.intended_moves[agent_id] = fallback
            end
        end
    end
end

# ============================================================================
# SECTION 9: EXECUTE MOVES SYSTEM
# ============================================================================

function system_execute_moves!(world, res::SimState)
    w, h = res.dims

    for (entities, gp, fp, vid, vm, ts) in Query(world, (GridPos, FracPos, VehicleId, VehicleMotion, TurnState))
        for i in eachindex(entities)
            eid = entities[i]._id
            if !res.intended_valid[eid]; continue; end

            g = gp[i]; f = fp[i]; v = vid[i]; m = vm[i]; t = ts[i]
            intended = res.intended_moves[eid]

            # Turn completion
            new_dir = v.direction
            new_ts_state = t.turning_state
            new_tp = t.turn_progress
            new_speed = m.speed
            new_tl_idx = v.traffic_light_idx

            if t.turning_state == TS_TURNING && t.turn_progress >= Float32(1.0)
                new_dir = get_post_turn_direction(v.direction)
                new_ts_state = TS_NOT_TURNING
                new_tp = Float32(0.0)
                new_speed = min(m.original_speed, m.speed * Float32(1.67))
                new_tl_idx = (new_dir == DIR_UP || new_dir == DIR_DOWN) ?
                    res.vertical_light_entity._id : res.horizontal_light_entity._id
            end

            # Fractional position handling
            new_fx = f.x; new_fy = f.y
            planned_gx = Int32(mod1(round(Int, f.x), w))
            planned_gy = Int32(mod1(round(Int, f.y), h))

            if intended != (planned_gx, planned_gy)
                dx = abs(Int(intended[1]) - round(Int, f.x))
                dy = abs(Int(intended[2]) - round(Int, f.y))
                is_wrapped = (dx > w - 10) || (dy > h - 10)
                if !is_wrapped
                    new_fx = Float64(intended[1])
                    new_fy = Float64(intended[2])
                end
            end

            new_gx = intended[1]; new_gy = intended[2]

            # Clear old occupied positions (inline, zero-alloc)
            old_vlen = get_vehicle_length(v.vehicle_type)
            old_dx, old_dy = pos_offset(v.direction)
            old_hx, old_hy = Int(g.x), Int(g.y)
            for k in 0:old_vlen-1
                cx = old_hx + k * old_dx; cy = old_hy + k * old_dy
                wx = mod1(cx, w); wy = mod1(cy, h)
                @inbounds if res.occupancy_grid[wx, wy] == eid
                    res.occupancy_grid[wx, wy] = UInt32(0)
                end
            end
            # Set new occupied positions (inline, zero-alloc)
            new_vlen = get_vehicle_length(v.vehicle_type)
            new_dx, new_dy = pos_offset(new_dir)
            new_hx, new_hy = Int(new_gx), Int(new_gy)
            for k in 0:new_vlen-1
                cx = new_hx + k * new_dx; cy = new_hy + k * new_dy
                wx = mod1(cx, w); wy = mod1(cy, h)
                @inbounds res.occupancy_grid[wx, wy] = eid
            end

            # Update in_intersection
            new_in_inter = is_in_intersection_ecs(Int(new_gx), Int(new_gy), new_dir, res)

            # Write back
            gp[i] = GridPos(new_gx, new_gy)
            fp[i] = FracPos(new_fx, new_fy)
            vid[i] = VehicleId(v.vehicle_type, new_dir, v.weight, v.planning_distance, new_tl_idx)
            vm[i] = VehicleMotion(new_speed, m.original_speed, m.previous_speed)
            ts[i] = TurnState(new_ts_state, t.turn_intention, new_tp, new_in_inter)
        end
    end

    # Emission decay (fused single-pass: x *= (1 - rate), clamped to 0)
    brake_retain = 1.0 - res.decay_rate * res.brake_decay_rate * res.dt
    tyre_retain = 1.0 - res.decay_rate * res.tyre_decay_rate * res.dt
    be = res.brake_emissions; te = res.tyre_emissions
    @inbounds @simd for i in eachindex(be)
        be[i] = max(0.0, be[i] * brake_retain)
    end
    @inbounds @simd for i in eachindex(te)
        te[i] = max(0.0, te[i] * tyre_retain)
    end
end

# ============================================================================
# SECTION 10: MAIN STEP FUNCTION
# ============================================================================

function ecs_model_step!(world)
    res = get_resource(world, SimState)
    system_traffic_lights!(world, res)
    system_plan_moves!(world, res)
    system_resolve_conflicts!(world, res)
    system_execute_moves!(world, res)
end

# ============================================================================
# SECTION 11: INITIALIZATION
# ============================================================================

function ecs_add_random_vehicle!(world, res::SimState, direction::UInt8, added_positions, vehicle_type::UInt8)
    dims = res.dims; sp = res.scaled_pos; w, h = dims; rng = res.rng

    spawn_range_outer = round(Int, w * 0.4)
    spawn_range_inner = w - spawn_range_outer

    local start_pos, occupied_positions
    max_spawn_attempts = 1000

    for attempt in 1:max_spawn_attempts
        start_pos = if direction == DIR_RIGHT
            (rand(rng, 1:spawn_range_outer), rand(rng, [sp.right_lane_right, sp.left_lane_right]))
        elseif direction == DIR_LEFT
            (rand(rng, spawn_range_inner:w), rand(rng, [sp.left_lane_left, sp.right_lane_left]))
        elseif direction == DIR_UP
            (rand(rng, [sp.left_lane_up, sp.right_lane_up]), rand(rng, 1:spawn_range_outer))
        else  # DIR_DOWN
            (rand(rng, [sp.left_lane_down, sp.right_lane_down]), rand(rng, spawn_range_inner:h))
        end

        occupied_positions = calculate_occupied_positions_ecs(vehicle_type, direction, start_pos)
        if !any(p -> p in added_positions, occupied_positions)
            break
        end
        if attempt == max_spawn_attempts
            error("Could not place vehicle after $max_spawn_attempts attempts — grid too crowded")
        end
    end

    tl_entity = (direction == DIR_UP || direction == DIR_DOWN) ?
        res.vertical_light_entity : res.horizontal_light_entity

    speed = generate_speed_ecs(res)
    if vehicle_type == VT_BUS; speed *= Float32(0.85); end

    weight = Float32(if vehicle_type == VT_SMART_CAR; 1000.0
        elseif vehicle_type == VT_CAR; 1500.0
        elseif vehicle_type == VT_E_CAR; 1700.0
        elseif vehicle_type == VT_SUV; 2200.0
        elseif vehicle_type == VT_E_SUV || vehicle_type == VT_OBSERVER; 2600.0
        else; res.bus_base_weight + res.bus_occupancy * 75.0; end)

    plan_dist = Int16(clamp(rand(rng, Poisson(res.planning_distance)) + 1, 1, 20))

    turn_intention = if direction == DIR_UP && start_pos[1] == sp.left_lane_up
        TI_LEFT
    elseif direction == DIR_DOWN && start_pos[1] == sp.left_lane_down
        rand(rng) < 0.2 ? TI_LEFT : TI_STRAIGHT
    elseif direction == DIR_LEFT && start_pos[2] == sp.left_lane_left
        rand(rng) < 0.2 ? TI_LEFT : TI_STRAIGHT
    elseif direction == DIR_RIGHT && start_pos[2] == sp.left_lane_right
        rand(rng) < 0.2 ? TI_LEFT : TI_STRAIGHT
    else
        TI_STRAIGHT
    end

    entity = new_entity!(world, (
        GridPos(Int32(start_pos[1]), Int32(start_pos[2])),
        FracPos(Float64(start_pos[1]), Float64(start_pos[2])),
        VehicleId(vehicle_type, direction, weight, plan_dist, tl_entity._id),
        VehicleMotion(speed, speed, speed),
        TurnState(TS_NOT_TURNING, turn_intention, Float32(0.0), false)
    ))

    # Register in id_to_entity
    eid = entity._id
    if eid > length(res.id_to_entity)
        resize!(res.id_to_entity, max(Int(eid) * 2, 256))
    end
    res.id_to_entity[eid] = entity

    # Register in occupancy grid
    for (cx, cy) in occupied_positions
        wx = mod1(cx, w); wy = mod1(cy, h)
        @inbounds res.occupancy_grid[wx, wy] = eid
    end

    for pos in occupied_positions
        push!(added_positions, pos)
    end
    return entity
end

function ecs_add_vehicle_at!(world, res::SimState, direction::UInt8, added_positions,
                             vehicle_type::UInt8, start_pos::Tuple{Int,Int}, veh_rng::AbstractRNG)
    dims = res.dims; sp = res.scaled_pos; w, h = dims

    occupied_positions = calculate_occupied_positions_ecs(vehicle_type, direction, start_pos)

    tl_entity = (direction == DIR_UP || direction == DIR_DOWN) ?
        res.vertical_light_entity : res.horizontal_light_entity

    speed = Float32(max(0.5, randn(veh_rng) * res.speed_variability + res.mean_speed))
    if vehicle_type == VT_BUS; speed *= Float32(0.85); end

    weight = Float32(if vehicle_type == VT_SMART_CAR; 1000.0
        elseif vehicle_type == VT_CAR; 1500.0
        elseif vehicle_type == VT_E_CAR; 1700.0
        elseif vehicle_type == VT_SUV; 2200.0
        elseif vehicle_type == VT_E_SUV || vehicle_type == VT_OBSERVER; 2600.0
        else; res.bus_base_weight + res.bus_occupancy * 75.0; end)

    plan_dist = Int16(clamp(rand(veh_rng, Poisson(res.planning_distance)) + 1, 1, 20))

    turn_intention = if direction == DIR_UP && start_pos[1] == sp.left_lane_up
        TI_LEFT
    elseif direction == DIR_DOWN && start_pos[1] == sp.left_lane_down
        rand(veh_rng) < 0.2 ? TI_LEFT : TI_STRAIGHT
    elseif direction == DIR_LEFT && start_pos[2] == sp.left_lane_left
        rand(veh_rng) < 0.2 ? TI_LEFT : TI_STRAIGHT
    elseif direction == DIR_RIGHT && start_pos[2] == sp.left_lane_right
        rand(veh_rng) < 0.2 ? TI_LEFT : TI_STRAIGHT
    else
        TI_STRAIGHT
    end

    entity = new_entity!(world, (
        GridPos(Int32(start_pos[1]), Int32(start_pos[2])),
        FracPos(Float64(start_pos[1]), Float64(start_pos[2])),
        VehicleId(vehicle_type, direction, weight, plan_dist, tl_entity._id),
        VehicleMotion(speed, speed, speed),
        TurnState(TS_NOT_TURNING, turn_intention, Float32(0.0), false)
    ))

    eid = entity._id
    if eid > length(res.id_to_entity)
        resize!(res.id_to_entity, max(Int(eid) * 2, 256))
    end
    res.id_to_entity[eid] = entity

    for (cx, cy) in occupied_positions
        wx = mod1(cx, w); wy = mod1(cy, h)
        @inbounds res.occupancy_grid[wx, wy] = eid
    end

    for pos in occupied_positions
        push!(added_positions, pos)
    end
    return entity
end

"""
Spawn fleet with deterministic positions per size class.
Each size class (2-cell, 3-cell, 4-cell, 7-cell) gets its own position pool
seeded from (base_seed + direction + size_offset). Changing the count of one
size class does NOT affect positions of other size classes.
Within a size class, swapping types (car↔e-car, SUV↔e-SUV) keeps the same position.
"""
function ecs_spawn_fleet!(world, res::SimState, dir::UInt8, base_seed::UInt64,
                          added_positions,
                          n_smart::Int, n_car::Int, n_ecar::Int,
                          n_esuv::Int, n_suv::Int, n_bus::Int)
    dims = res.dims; sp = res.scaled_pos; w, h = dims
    spawn_range_outer = round(Int, w * 0.4)
    spawn_range_inner = w - spawn_range_outer

    function gen_pos(rng)
        if dir == DIR_RIGHT
            (rand(rng, 1:spawn_range_outer), rand(rng, [sp.right_lane_right, sp.left_lane_right]))
        elseif dir == DIR_LEFT
            (rand(rng, spawn_range_inner:w), rand(rng, [sp.left_lane_left, sp.right_lane_left]))
        elseif dir == DIR_UP
            (rand(rng, [sp.left_lane_up, sp.right_lane_up]), rand(rng, 1:spawn_range_outer))
        else
            (rand(rng, [sp.left_lane_down, sp.right_lane_down]), rand(rng, spawn_range_inner:h))
        end
    end

    # One position pool per size class, each with its own RNG
    dir_idx = UInt64(dir)
    function make_pool(size_offset::UInt64, pool_size::Int)
        rng = Random.MersenneTwister(base_seed + dir_idx * 100 + size_offset)
        [gen_pos(rng) for _ in 1:pool_size]
    end

    micro_pool = make_pool(UInt64(10), 50)   # 2-cell: microcars
    car_pool   = make_pool(UInt64(20), 100)  # 3-cell: cars, e-cars
    suv_pool   = make_pool(UInt64(30), 100)  # 4-cell: SUVs, e-SUVs
    bus_pool   = make_pool(UInt64(40), 100)  # 7-cell: buses (large pool for dense fleets)

    function place_from_pool!(pool, vehicles::Vector{UInt8})
        # Each vehicle also gets a per-slot RNG for speed/planning/turn
        slot_base = base_seed + dir_idx * 1000
        cand_idx = 1
        for (slot, vtype) in enumerate(vehicles)
            slot_rng = Random.MersenneTwister(slot_base + UInt64(slot))
            placed = false
            while cand_idx <= length(pool) && !placed
                pos = pool[cand_idx]
                cand_idx += 1
                occ = calculate_occupied_positions_ecs(vtype, dir, pos)
                if !any(p -> p in added_positions, occ)
                    ecs_add_vehicle_at!(world, res, dir, added_positions, vtype, pos, slot_rng)
                    placed = true
                end
            end
            if !placed
                @warn "Could not place $(VTYPE_TO_SYM[vtype]) in dir $(DIR_TO_SYM[dir]) — pool exhausted"
            end
        end
    end

    # Place each size class from its own pool
    # Within a pool, e-variants come first so slot positions are stable when swapping ICE↔EV
    # Microcars placed from car_pool (after e-cars and remaining cars) so they inherit
    # the exact positions of the cars they replace — ensures clean in-place swapping
    place_from_pool!(car_pool,   vcat(fill(VT_E_CAR, n_ecar), fill(VT_CAR, n_car), fill(VT_SMART_CAR, n_smart)))
    place_from_pool!(suv_pool,   vcat(fill(VT_E_SUV, n_esuv), fill(VT_SUV, n_suv)))
    place_from_pool!(bus_pool,   fill(VT_BUS, n_bus))
end

function ecs_initialise_model(;
    dims = try (parse(Int, get(ENV, "GRID_SIZE", "200")), parse(Int, get(ENV, "GRID_SIZE", "200"))) catch; (200, 200) end,
    n_smart_cars_left = 0, n_smart_cars_right = 0, n_smart_cars_up = 0, n_smart_cars_down = 0,
    n_cars_left = 0, n_cars_right = 0, n_cars_up = 0, n_cars_down = 0,
    n_e_cars_left = 0, n_e_cars_right = 0, n_e_cars_up = 0, n_e_cars_down = 0,
    n_suvs_left = 0, n_suvs_right = 0, n_suvs_up = 0, n_suvs_down = 0,
    n_buses_left = 0, n_buses_right = 0, n_buses_up = 0, n_buses_down = 0,
    n_e_suvs_left = 0, n_e_suvs_right = 0, n_e_suvs_up = 0, n_e_suvs_down = 0,
    decay_rate = 0.01,
    brake_decay_rate = 1.0, tyre_decay_rate = 1.0,
    horizontal_stop_right = nothing, horizontal_stop_left = nothing,
    vertical_stop_up = nothing, vertical_stop_down = nothing,
    mean_speed = 5.0, speed_variability = 1.0,
    max_sight_distance = 10, planning_distance = 5.0,
    regen_eff::Float64 = REGEN_EFF_ECS,
    rng::Union{AbstractRNG,Nothing} = nothing,
    green_duration_s::Float64 = DEFAULT_GREEN_S,
    amber_duration_s::Float64 = DEFAULT_AMBER_S,
    red_duration_s::Float64 = DEFAULT_RED_S,
    brake_dispersion_radius::Int = 3, tyre_dispersion_radius::Int = 3,
    bus_base_weight::Float64 = 12000.0, bus_occupancy::Float64 = 22.5,
    spawn_observer::Bool = true,  # Spawn dedicated observer vehicle (VT_OBSERVER)
    world = nothing  # Optional: pass existing world for reset! reuse
)
    if rng === nothing
        rng = Random.MersenneTwister(12345)
    end

    # Create or reset world
    if world === nothing
        world = World(
            GridPos => Storage{StructArray},
            FracPos => Storage{StructArray},
            VehicleId => Storage{StructArray},
            VehicleMotion => Storage{StructArray},
            TurnState => Storage{StructArray},
            TrafficLightComp => Storage{StructArray};
            initial_capacity = 256
        )
    else
        reset!(world)
    end

    scaled_pos = get_scaled_positions_ecs(dims)
    h_stop_right = horizontal_stop_right !== nothing ? horizontal_stop_right : scaled_pos.bottom_wall
    h_stop_left = horizontal_stop_left !== nothing ? horizontal_stop_left : scaled_pos.top_wall
    v_stop_up = vertical_stop_up !== nothing ? vertical_stop_up : scaled_pos.bottom_wall
    v_stop_down = vertical_stop_down !== nothing ? vertical_stop_down : scaled_pos.top_wall
    dt = calculate_adaptive_dt_ecs(mean_speed, speed_variability=speed_variability)

    ze = Ark.zero_entity
    max_ents = 256

    res = SimState(
        zeros(dims...), zeros(dims...), zeros(Bool, dims...), zeros(UInt32, dims...),
        fill(ze, max_ents),
        Vector{NTuple{2,Int32}}(undef, max_ents), falses(max_ents),
        zeros(UInt8, max_ents), zeros(UInt8, max_ents),
        dims, decay_rate, brake_decay_rate, tyre_decay_rate,
        h_stop_right, h_stop_left, v_stop_up, v_stop_down,
        mean_speed, speed_variability, max_sight_distance, planning_distance,
        dt, regen_eff,
        green_duration_s, amber_duration_s, red_duration_s,
        brake_dispersion_radius, tyre_dispersion_radius,
        bus_base_weight, bus_occupancy,
        ze, ze,   # light entities (set below)
        TL_GREEN, TL_RED,  # cached light states
        rng, scaled_pos,
        Vector{Tuple{Int,Int}}(undef, 7),
        # _cache_* vectors (indexed by entity id)
        zeros(UInt8, max_ents),    # _cache_dir
        zeros(UInt8, max_ents),    # _cache_vtype
        zeros(Float64, max_ents),  # _cache_fx
        zeros(Float64, max_ents),  # _cache_fy
        zeros(Int32, max_ents),    # _cache_gx
        zeros(Int32, max_ents),    # _cache_gy
        zeros(Float32, max_ents),  # _cache_speed
        zeros(Float32, max_ents),  # _cache_orig_speed
        zeros(UInt8, max_ents),    # _cache_ts
        zeros(Int8, max_ents),     # _cache_vlen
        zeros(Float32, max_ents),  # _cache_weight
        # Pre-allocated conflict detection
        zeros(UInt32, dims...),    # _intended_occ
        zeros(UInt32, max_ents),   # _conflict_eids
        zeros(Float64, max_ents),  # _conflict_prio
    )

    # Buildings (same as original)
    width, height = dims
    res.buildings[vcat(1:scaled_pos.bottom_wall, scaled_pos.top_wall:height), scaled_pos.bottom_wall] .= true
    res.buildings[vcat(1:scaled_pos.bottom_wall, scaled_pos.top_wall:height), scaled_pos.top_wall] .= true
    res.buildings[scaled_pos.left_wall, vcat(1:scaled_pos.bottom_wall, scaled_pos.top_wall:width)] .= true
    res.buildings[scaled_pos.right_wall, vcat(1:scaled_pos.bottom_wall, scaled_pos.top_wall:width)] .= true

    center_x, center_y = scaled_pos.intersection_center

    h_light = new_entity!(world, (
        GridPos(Int32(center_x), Int32(center_y + 1)),
        TrafficLightComp(TL_GREEN, LT_HORIZONTAL, Int32(0))
    ))
    v_light = new_entity!(world, (
        GridPos(Int32(center_x + 1), Int32(center_y)),
        TrafficLightComp(TL_RED, LT_VERTICAL, Int32(0))
    ))

    res.horizontal_light_entity = h_light
    res.vertical_light_entity = v_light
    res.id_to_entity[h_light._id] = h_light
    res.id_to_entity[v_light._id] = v_light

    added_positions = Set{Tuple{Int,Int}}()

    # Generate a base seed for deterministic fleet spawning (from main RNG)
    fleet_base_seed = rand(rng, UInt64)

    # Observer spawns FIRST at a fixed position — before any fleet vehicles
    if spawn_observer
        obs_x = round(Int, dims[1] * 0.9)  # 90% along grid (=180 for 200×200)
        obs_y = res.scaled_pos.right_lane_left
        obs_pos = (obs_x, obs_y)
        obs_occ = calculate_occupied_positions_ecs(VT_OBSERVER, DIR_LEFT, obs_pos)
        while any(p -> p in added_positions, obs_occ)
            obs_x -= 5
            obs_pos = (obs_x, obs_y)
            obs_occ = calculate_occupied_positions_ecs(VT_OBSERVER, DIR_LEFT, obs_pos)
        end
        obs_rng = Random.MersenneTwister(fleet_base_seed + UInt64(999))
        ecs_add_vehicle_at!(world, res, DIR_LEFT, added_positions, VT_OBSERVER, obs_pos, obs_rng)
    end

    # Spawn fleet with deterministic per-size-class position pools
    for (dir, n_smart, n_car, n_ecar, n_esuv, n_suv, n_bus) in [
        (DIR_LEFT,  n_smart_cars_left,  n_cars_left,  n_e_cars_left,  n_e_suvs_left,  n_suvs_left,  n_buses_left),
        (DIR_RIGHT, n_smart_cars_right, n_cars_right, n_e_cars_right, n_e_suvs_right, n_suvs_right, n_buses_right),
        (DIR_UP,    n_smart_cars_up,    n_cars_up,    n_e_cars_up,    n_e_suvs_up,    n_suvs_up,    n_buses_up),
        (DIR_DOWN,  n_smart_cars_down,  n_cars_down,  n_e_cars_down,  n_e_suvs_down,  n_suvs_down,  n_buses_down),
    ]
        # For DIR_LEFT, subtract 1 e-SUV since observer takes its slot
        actual_esuv = (dir == DIR_LEFT && spawn_observer) ? max(n_esuv - 1, 0) : n_esuv
        ecs_spawn_fleet!(world, res, dir, fleet_base_seed, added_positions,
                         n_smart, n_car, n_ecar, actual_esuv, n_suv, n_bus)
    end

    # Resize buffers to accommodate all entities
    max_eid = 0
    for (ents,) in Query(world, (GridPos,))
        for i in eachindex(ents)
            max_eid = max(max_eid, Int(ents[i]._id))
        end
    end
    max_eid = max(max_eid + 16, 256)
    resize!(res.id_to_entity, max_eid)
    resize!(res.intended_moves, max_eid)
    resize!(res.intended_valid, max_eid)
    resize!(res.intended_vehicle_type, max_eid)
    resize!(res.intended_direction, max_eid)
    resize!(res._cache_dir, max_eid)
    resize!(res._cache_vtype, max_eid)
    resize!(res._cache_fx, max_eid)
    resize!(res._cache_fy, max_eid)
    resize!(res._cache_gx, max_eid)
    resize!(res._cache_gy, max_eid)
    resize!(res._cache_speed, max_eid)
    resize!(res._cache_orig_speed, max_eid)
    resize!(res._cache_ts, max_eid)
    resize!(res._cache_vlen, max_eid)
    resize!(res._cache_weight, max_eid)
    resize!(res._conflict_eids, max_eid)
    resize!(res._conflict_prio, max_eid)

    add_resource!(world, res)
    return world
end

# ============================================================================
# SECTION 12: BACKWARD-COMPATIBLE run_simulation API
# ============================================================================

# Simple DataFrame-like wrapper for backward compatibility
struct SimpleModelDF
    columns::Dict{Symbol, Vector}
end
Base.getindex(sdf::SimpleModelDF, ::typeof(!), key::Symbol) = sdf.columns[key]

function run_simulation(;
    dims::Tuple{Int,Int} = try (parse(Int, get(ENV, "GRID_SIZE", "200")), parse(Int, get(ENV, "GRID_SIZE", "200"))) catch; (200, 200) end,
    n_smart_cars_left = 0, n_smart_cars_right = 0, n_smart_cars_up = 0, n_smart_cars_down = 0,
    n_cars_left = 10, n_cars_right = 10, n_cars_up = 10, n_cars_down = 10,
    n_e_cars_left = 1, n_e_cars_right = 1, n_e_cars_up = 1, n_e_cars_down = 1,
    n_suvs_left = 12, n_suvs_right = 12, n_suvs_up = 12, n_suvs_down = 12,
    n_e_suvs_left = 1, n_e_suvs_right = 1, n_e_suvs_up = 1, n_e_suvs_down = 1,
    n_buses_left = 1, n_buses_right = 1, n_buses_up = 1, n_buses_down = 1,
    mean_speed = 5.0, speed_variability = 1.0,
    brake_decay_rate = 1.0, tyre_decay_rate = 2.0,
    max_sight_distance = 10, planning_distance = 5.0,
    static_observer_locations = Dict("corner" => (92, 90)),
    tracked_vehicle_type::Union{Symbol,Nothing} = :observer,
    warn_if_missing_tracked::Bool = true,
    steps = 1000,
    data_sample_interval::Int = 1,
    random_seed::Union{Int,Nothing} = nothing,
    regen_eff::Float64 = REGEN_EFF_ECS,
    green_duration_s::Float64 = DEFAULT_GREEN_S,
    amber_duration_s::Float64 = DEFAULT_AMBER_S,
    red_duration_s::Float64 = DEFAULT_RED_S,
    brake_dispersion_radius::Int = 3, tyre_dispersion_radius::Int = 3,
    bus_base_weight::Float64 = 12000.0, bus_occupancy::Float64 = 22.5,
    fast_mode::Bool = false,
    grid_warmup_steps::Int = 0
)
    local_rng = (random_seed === nothing) ? nothing : Random.MersenneTwister(random_seed)

    world = ecs_initialise_model(;
        dims=dims, rng=local_rng,
        n_smart_cars_left, n_smart_cars_right, n_smart_cars_up, n_smart_cars_down,
        n_cars_left, n_cars_right, n_cars_up, n_cars_down,
        n_e_cars_left, n_e_cars_right, n_e_cars_up, n_e_cars_down,
        n_suvs_left, n_suvs_right, n_suvs_up, n_suvs_down,
        n_e_suvs_left, n_e_suvs_right, n_e_suvs_up, n_e_suvs_down,
        n_buses_left, n_buses_right, n_buses_up, n_buses_down,
        mean_speed, speed_variability,
        brake_decay_rate, tyre_decay_rate,
        max_sight_distance, planning_distance,
        regen_eff, green_duration_s, amber_duration_s, red_duration_s,
        brake_dispersion_radius, tyre_dispersion_radius,
        bus_base_weight, bus_occupancy,
        spawn_observer = (tracked_vehicle_type === :observer)
    )

    res = get_resource(world, SimState)

    # === FAST MODE ===
    if fast_mode && tracked_vehicle_type !== nothing
        tracked_vtype = SYM_TO_VTYPE[tracked_vehicle_type]
        tracked_eids = UInt32[]

        for (entities, vid_col) in Query(world, (VehicleId,))
            for i in eachindex(entities)
                if vid_col[i].vehicle_type == tracked_vtype
                    push!(tracked_eids, entities[i]._id)
                end
            end
        end
        sort!(tracked_eids)

        if isempty(tracked_eids)
            if warn_if_missing_tracked
                @warn "Fast mode: No vehicles of type $tracked_vehicle_type found"
            end
            return nothing, Dict(
                "all_tracked_vehicles" => Dict(
                    :brake_emissions => Vector{Float64}[],
                    :tyre_emissions => Vector{Float64}[],
                    :speeds => Vector{Float64}[],
                    :n_vehicles => 0
                )
            )
        end

        n_samples = steps ÷ data_sample_interval + 2
        all_brake = [Vector{Float64}(undef, n_samples) for _ in tracked_eids]
        all_tyre = [Vector{Float64}(undef, n_samples) for _ in tracked_eids]
        all_speeds = [Vector{Float64}(undef, n_samples) for _ in tracked_eids]

        brake_accum = zeros(dims...)
        tyre_accum = zeros(dims...)
        n_accum = 0
        sample_idx = 1

        # Collect at step 0
        for (v_idx, eid) in enumerate(tracked_eids)
            entity = res.id_to_entity[eid]
            gpos, vmot = get_components(world, entity, (GridPos, VehicleMotion))
            x, y = Int(gpos.x), Int(gpos.y)
            all_brake[v_idx][sample_idx] = res.brake_emissions[x, y]
            all_tyre[v_idx][sample_idx] = res.tyre_emissions[x, y]
            all_speeds[v_idx][sample_idx] = Float64(vmot.speed)
        end
        sample_idx += 1

        for step_num in 1:steps
            ecs_model_step!(world)

            if step_num > grid_warmup_steps
                brake_accum .+= res.brake_emissions
                tyre_accum .+= res.tyre_emissions
                n_accum += 1
            end

            if step_num % data_sample_interval == 0
                for (v_idx, eid) in enumerate(tracked_eids)
                    entity = res.id_to_entity[eid]
                    gpos, vmot = get_components(world, entity, (GridPos, VehicleMotion))
                    x, y = Int(gpos.x), Int(gpos.y)
                    all_brake[v_idx][sample_idx] = res.brake_emissions[x, y]
                    all_tyre[v_idx][sample_idx] = res.tyre_emissions[x, y]
                    all_speeds[v_idx][sample_idx] = Float64(vmot.speed)
                end
                sample_idx += 1
            end
        end

        avg_brake_grid = brake_accum ./ max(n_accum, 1)
        avg_tyre_grid = tyre_accum ./ max(n_accum, 1)

        for v_idx in 1:length(tracked_eids)
            resize!(all_brake[v_idx], sample_idx - 1)
            resize!(all_tyre[v_idx], sample_idx - 1)
            resize!(all_speeds[v_idx], sample_idx - 1)
        end

        observer_data = Dict(
            "all_tracked_vehicles" => Dict(
                :brake_emissions => all_brake,
                :tyre_emissions => all_tyre,
                :speeds => all_speeds,
                :n_vehicles => length(tracked_eids)
            ),
            "final_brake_grid" => copy(res.brake_emissions),
            "final_tyre_grid" => copy(res.tyre_emissions),
            "avg_brake_grid" => avg_brake_grid,
            "avg_tyre_grid" => avg_tyre_grid
        )
        return nothing, observer_data
    end

    # === FULL MODE ===
    function collect_car_data_ecs(world, res)
        data = Tuple{UInt32, Symbol, Tuple{Int,Int}, Symbol, Float64}[]
        for (entities, gpos, vid_col, vmot) in Query(world, (GridPos, VehicleId, VehicleMotion))
            for i in eachindex(entities)
                push!(data, (
                    entities[i]._id,
                    VTYPE_TO_SYM[vid_col[i].vehicle_type],
                    (Int(gpos[i].x), Int(gpos[i].y)),
                    DIR_TO_SYM[vid_col[i].direction],
                    Float64(vmot[i].speed)
                ))
            end
        end
        sort!(data, by = d -> d[1])
        return data
    end

    all_car_data = Vector{Vector{Tuple{UInt32, Symbol, Tuple{Int,Int}, Symbol, Float64}}}()
    all_brake_grids = Vector{Matrix{Float64}}()
    all_tyre_grids = Vector{Matrix{Float64}}()

    # Step 0
    push!(all_car_data, collect_car_data_ecs(world, res))
    push!(all_brake_grids, copy(res.brake_emissions))
    push!(all_tyre_grids, copy(res.tyre_emissions))

    for step_num in 1:steps
        ecs_model_step!(world)
        if step_num % data_sample_interval == 0
            push!(all_car_data, collect_car_data_ecs(world, res))
            push!(all_brake_grids, copy(res.brake_emissions))
            push!(all_tyre_grids, copy(res.tyre_emissions))
        end
    end

    car_data_per_step = all_car_data
    brake_emissions = all_brake_grids
    tyre_emissions = all_tyre_grids

    # Track vehicles
    tracked_agent_ids = UInt32[]
    if tracked_vehicle_type !== nothing
        tracked_vtype_sym = tracked_vehicle_type
        for (id, vt, pos, dir, speed) in car_data_per_step[1]
            if vt == tracked_vtype_sym
                push!(tracked_agent_ids, id)
            end
        end
        if isempty(tracked_agent_ids)
            if warn_if_missing_tracked
                @warn "Could not find any vehicles of type $tracked_vehicle_type to track."
            end
            nan_vec = fill(NaN, length(car_data_per_step))
            static_observers_data = Dict{String, Dict{Symbol, Vector{Float64}}}()
            for (name, location) in static_observer_locations
                static_observers_data[name] = Dict(
                    :brake_emissions => [brake_emissions[i][location...] for i in 1:length(brake_emissions)],
                    :tyre_emissions => [tyre_emissions[i][location...] for i in 1:length(tyre_emissions)]
                )
            end
            model_df = SimpleModelDF(Dict(
                :collect_car_data => all_car_data,
                :collect_brake_emissions => all_brake_grids,
                :collect_tyre_emissions => all_tyre_grids
            ))
            observer_data = Dict(
                "moving_observer" => Dict(:speed => nan_vec, :brake_emissions => nan_vec, :tyre_emissions => nan_vec),
                "static_observers" => static_observers_data,
                "final_brake_grid" => copy(res.brake_emissions),
                "final_tyre_grid" => copy(res.tyre_emissions)
            )
            return model_df, observer_data
        end
    end

    all_vehicles_brake = [Float64[] for _ in tracked_agent_ids]
    all_vehicles_tyre = [Float64[] for _ in tracked_agent_ids]
    all_vehicles_speeds = [Float64[] for _ in tracked_agent_ids]

    if tracked_vehicle_type !== nothing
        for i in 1:length(car_data_per_step)
            step_data = car_data_per_step[i]
            for (vehicle_idx, agent_id) in enumerate(tracked_agent_ids)
                tracked_info = findfirst(d -> d[1] == agent_id, step_data)
                if isnothing(tracked_info)
                    push!(all_vehicles_speeds[vehicle_idx], NaN)
                    push!(all_vehicles_brake[vehicle_idx], NaN)
                    push!(all_vehicles_tyre[vehicle_idx], NaN)
                else
                    _, _, pos, _, speed = step_data[tracked_info]
                    x, y = pos
                    push!(all_vehicles_speeds[vehicle_idx], speed)
                    push!(all_vehicles_brake[vehicle_idx], brake_emissions[i][x, y])
                    push!(all_vehicles_tyre[vehicle_idx], tyre_emissions[i][x, y])
                end
            end
        end
        moving_observer_brake = vcat(all_vehicles_brake...)
        moving_observer_tyre = vcat(all_vehicles_tyre...)
        moving_observer_speeds = vcat(all_vehicles_speeds...)
    else
        moving_observer_speeds = fill(NaN, length(car_data_per_step))
        moving_observer_brake = fill(NaN, length(car_data_per_step))
        moving_observer_tyre = fill(NaN, length(car_data_per_step))
        all_vehicles_brake = Vector{Float64}[]
        all_vehicles_tyre = Vector{Float64}[]
        all_vehicles_speeds = Vector{Float64}[]
    end

    static_observers_data = Dict{String, Dict{Symbol, Vector{Float64}}}()
    for (name, location) in static_observer_locations
        static_observers_data[name] = Dict(
            :brake_emissions => [brake_emissions[i][location...] for i in 1:length(brake_emissions)],
            :tyre_emissions => [tyre_emissions[i][location...] for i in 1:length(tyre_emissions)]
        )
    end

    model_df = SimpleModelDF(Dict(
        :collect_car_data => all_car_data,
        :collect_brake_emissions => all_brake_grids,
        :collect_tyre_emissions => all_tyre_grids
    ))

    observer_data = Dict(
        "moving_observer" => Dict(
            :speed => moving_observer_speeds,
            :brake_emissions => moving_observer_brake,
            :tyre_emissions => moving_observer_tyre
        ),
        "all_tracked_vehicles" => Dict(
            :brake_emissions => all_vehicles_brake,
            :tyre_emissions => all_vehicles_tyre,
            :speeds => all_vehicles_speeds,
            :n_vehicles => length(tracked_agent_ids)
        ),
        "static_observers" => static_observers_data,
        "final_brake_grid" => copy(res.brake_emissions),
        "final_tyre_grid" => copy(res.tyre_emissions)
    )

    return model_df, observer_data
end

# Re-export calculate_adaptive_dt for external scripts
const calculate_adaptive_dt = calculate_adaptive_dt_ecs
