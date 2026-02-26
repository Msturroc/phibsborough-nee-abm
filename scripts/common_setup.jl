"""
common_setup.jl — Shared setup for all figure scripts.

Loads the model, calibrated parameters, experimental data, and defines
the 4-seed repeat methodology used throughout.

Key change from v1: uses dedicated VT_OBSERVER vehicle type for moving
observer instead of piggybacking on the first e-SUV. The observer is
spawned before all fleet vehicles so its entity ID and RNG-determined
starting position are stable regardless of fleet composition.

Include this from any figure script:
    include("common_setup.jl")
"""

using Statistics
using DelimitedFiles
using Printf
using CSV
using DataFrames
using Random

const GRID_SIZE = 200
ENV["GRID_SIZE"] = string(GRID_SIZE)

# ============================================================================
# Load model (ECS version — bit-perfect, 2.8x faster step function)
# ============================================================================
const PROJECT_ROOT = joinpath(@__DIR__, "..")
println("Loading model from: $(joinpath(PROJECT_ROOT, "src", "phibsborough_ecs_v2.jl"))")
include(joinpath(PROJECT_ROOT, "src", "phibsborough_ecs_v2.jl"))

# Aliases so common_setup API matches original model names
const REGEN_EFF = REGEN_EFF_ECS
const calculate_adaptive_dt = calculate_adaptive_dt_ecs

# ============================================================================
# Load calibrated parameters (15 log10 params from CMA-ES v7)
# ============================================================================
const PARAM_FILE = joinpath(PROJECT_ROOT, "data", "best_parameters_cmaes_v7_excellent_contender.txt")
println("Loading parameters from: $PARAM_FILE")
const LOG10_PARAMS = vec(readdlm(PARAM_FILE))

const LOWER_BOUNDS = Float64[log10(8.0), log10(0.5), log10(0.001), log10(0.01), log10(0.3), log10(0.3), log10(0.5), log10(0.3), log10(10.0), log10(1.0), log10(15.0), 1.0, 1.0, -2000.0, -0.5]
const UPPER_BOUNDS = Float64[log10(17.5), log10(5.0), log10(1e7), log10(500.0), log10(5.0), log10(200.0), log10(5.0), log10(6.0), log10(20.0), log10(10.0), log10(90.0), 6.0, 6.0, 2000.0, 2.0]

const CLAMPED = [clamp(LOG10_PARAMS[i], LOWER_BOUNDS[i], UPPER_BOUNDS[i]) for i in 1:length(LOG10_PARAMS)]
const PARAMS = 10 .^ CLAMPED[1:11]

const MEAN_SPEED      = PARAMS[1]
const SPEED_VAR       = PARAMS[2]
const BRAKE_TO_UFP    = PARAMS[3]
const TYRE_TO_PM25    = PARAMS[4]
const BRAKE_DECAY     = PARAMS[5]
const TYRE_DECAY      = PARAMS[6]
const BRAKE_EXP       = PARAMS[7]
const TYRE_EXP        = PARAMS[8]
const SIGHT_DIST      = round(Int, PARAMS[9])
const PLANNING_DIST   = PARAMS[10]
const GREEN_DUR       = PARAMS[11]
const BRAKE_DISP      = clamp(round(Int, CLAMPED[12]), 1, 6)
const TYRE_DISP       = clamp(round(Int, CLAMPED[13]), 1, 6)
const OFFSET_UFP      = CLAMPED[14]
const OFFSET_PM25     = CLAMPED[15]

@printf("  mean_speed = %.2f m/s (%.1f km/h)\n", MEAN_SPEED, MEAN_SPEED * 3.6)
@printf("  speed_var  = %.4f m/s\n", SPEED_VAR)

# ============================================================================
# Load experimental data for baseline offsets
# ============================================================================
println("Loading experimental data...")
const LAT_MIN, LAT_MAX = 53.359899, 53.361701
const LON_MIN, LON_MAX = -6.27421, -6.27119
const _full_airview = CSV.read(joinpath(PROJECT_ROOT, "AirView_DublinCity_Measurements_ugm3.csv"), DataFrame)
const _phibs = filter(row -> (LAT_MIN <= row.latitude <= LAT_MAX) && (LON_MIN <= row.longitude <= LON_MAX), _full_airview)
const GOOGLE_PM25 = Float64.(dropmissing(_phibs, :PM25_ugm3).PM25_ugm3)
const _phibs_ufp = dropmissing(_phibs, [:PMch1_perL, :PMch2_perL])
const GOOGLE_UFP = Float64.(_phibs_ufp.PMch1_perL + _phibs_ufp.PMch2_perL)
const BASELINE_UFP  = minimum(GOOGLE_UFP)
const BASELINE_PM25 = minimum(GOOGLE_PM25)
@printf("  BASELINE_UFP  = %.1f #/L\n", BASELINE_UFP)
@printf("  BASELINE_PM25 = %.2f µg/m³\n", BASELINE_PM25)

# ============================================================================
# 4-seed repeat methodology
# ============================================================================
const SCENARIO_SEEDS = [92345, 22345, 62345]
const N_REPEATS      = 3
const TOTAL_TIME     = 1100.0  # seconds
const WARMUP_TIME    = 100.0   # seconds (first 100s discarded as burn-in)

println("  Repeats: $N_REPEATS, Seeds: $SCENARIO_SEEDS")
println("  Simulation: $(TOTAL_TIME)s total ($(WARMUP_TIME)s warmup)")

# ============================================================================
# Baseline fleet (per direction) — matches calibration
# ============================================================================
const N_CARS_PER_DIR  = 10
const N_ECARS_PER_DIR = 1
const N_SUVS_PER_DIR  = 12
const N_ESUVS_PER_DIR = 1
const N_BUSES_PER_DIR = 1

# Totals (all 4 directions)
const TOTAL_CARS  = N_CARS_PER_DIR * 4    # 40
const TOTAL_ECARS = N_ECARS_PER_DIR * 4   # 4
const TOTAL_SUVS  = N_SUVS_PER_DIR * 4    # 48
const TOTAL_ESUVS = N_ESUVS_PER_DIR * 4   # 4
const TOTAL_BUSES = N_BUSES_PER_DIR * 4    # 4
const TOTAL_PRIVATE = TOTAL_CARS + TOTAL_ECARS + TOTAL_SUVS + TOTAL_ESUVS  # 96
const TOTAL_CONVENTIONAL = TOTAL_CARS + TOTAL_SUVS  # 88

# Occupancy constants
const PASSENGERS_PER_CAR   = 1.27
const PASSENGERS_PER_MICRO = 1.2
const BUS_CAPACITY         = 90
const DEFAULT_BUS_OCC_FRAC = 0.25   # afternoon inter-peak
const SCENARIO_BUS_OCC_FRAC = 0.95  # scenario buses run near-full
const PASSENGERS_IN_PRIVATE = TOTAL_PRIVATE * PASSENGERS_PER_CAR
const DEFAULT_BUS_OCC       = BUS_CAPACITY * DEFAULT_BUS_OCC_FRAC

# ============================================================================
# Helper: run simulation and extract moving observer data
# ============================================================================
"""
    run_moving_observer(; mean_speed, speed_var, n_cars, n_ecars, n_suvs, n_esuvs,
                          n_buses, n_micros, bus_occ, regen_eff, seed)

Run a single simulation and return (ufp_timeseries, pm25_timeseries) for moving observer,
sampled at 1Hz after warmup.
"""
function run_moving_observer(;
    mean_speed::Float64 = MEAN_SPEED,
    speed_var::Float64   = SPEED_VAR,
    n_cars::Int          = N_CARS_PER_DIR,
    n_ecars::Int         = N_ECARS_PER_DIR,
    n_suvs::Int          = N_SUVS_PER_DIR,
    n_esuvs::Int         = N_ESUVS_PER_DIR,
    n_buses::Int         = N_BUSES_PER_DIR,
    n_micros::Int        = 0,
    bus_occ::Float64     = DEFAULT_BUS_OCC,
    regen_eff::Float64   = REGEN_EFF,
    seed::Int            = 12345
)
    dt = calculate_adaptive_dt(mean_speed, speed_variability=speed_var)
    n_steps = round(Int, TOTAL_TIME / dt)

    run_args = Dict{Symbol, Any}(
        :dims => (GRID_SIZE, GRID_SIZE),
        :mean_speed => mean_speed,
        :speed_variability => speed_var,
        :brake_decay_rate => BRAKE_DECAY,
        :tyre_decay_rate => TYRE_DECAY,
        :max_sight_distance => SIGHT_DIST,
        :planning_distance => PLANNING_DIST,
        :random_seed => seed,
        :green_duration_s => GREEN_DUR,
        :amber_duration_s => 3.0,
        :red_duration_s => GREEN_DUR + 3.0,
        :brake_dispersion_radius => BRAKE_DISP,
        :tyre_dispersion_radius => TYRE_DISP,
        :steps => n_steps,
        :data_sample_interval => max(1, round(Int, 1.0 / dt)),
        :tracked_vehicle_type => :observer,
        :bus_base_weight => 12000.0,
        :bus_occupancy => bus_occ,
        :regen_eff => regen_eff,
        # Fleet per direction (observer is extra, spawned by model with its own RNG)
        :n_cars_left => n_cars, :n_cars_right => n_cars,
        :n_cars_up => n_cars, :n_cars_down => n_cars,
        :n_e_cars_left => n_ecars, :n_e_cars_right => n_ecars,
        :n_e_cars_up => n_ecars, :n_e_cars_down => n_ecars,
        :n_suvs_left => n_suvs, :n_suvs_right => n_suvs,
        :n_suvs_up => n_suvs, :n_suvs_down => n_suvs,
        :n_e_suvs_left => n_esuvs, :n_e_suvs_right => n_esuvs,
        :n_e_suvs_up => n_esuvs, :n_e_suvs_down => n_esuvs,
        :n_buses_left => n_buses, :n_buses_right => n_buses,
        :n_buses_up => n_buses, :n_buses_down => n_buses,
        :n_smart_cars_left => n_micros, :n_smart_cars_right => n_micros,
        :n_smart_cars_up => n_micros, :n_smart_cars_down => n_micros,
    )

    _, observer_data = run_simulation(; run_args...)
    all_data = observer_data["all_tracked_vehicles"]
    brake_raw = all_data[:brake_emissions][1]
    tyre_raw  = all_data[:tyre_emissions][1]
    step_times = [(i-1)*dt*run_args[:data_sample_interval] for i in 1:length(brake_raw)]

    # Sample at 1Hz after warmup
    brake_sampled, tyre_sampled = Float64[], Float64[]
    for t in WARMUP_TIME:1.0:TOTAL_TIME
        idx_t = clamp(searchsortedlast(step_times, t), 1, length(step_times))
        push!(brake_sampled, brake_raw[idx_t])
        push!(tyre_sampled, tyre_raw[idx_t])
    end

    sim_ufp  = (BASELINE_UFP + OFFSET_UFP)   .+ BRAKE_TO_UFP .* (brake_sampled .^ BRAKE_EXP)
    sim_pm25 = (BASELINE_PM25 + OFFSET_PM25) .+ TYRE_TO_PM25 .* (tyre_sampled  .^ TYRE_EXP)

    return (ufp=sim_ufp, pm25=sim_pm25)
end

"""
    run_scenario_repeats(; kwargs...)

Run all 4 seeds for a scenario, returning combined (ufp, pm25) vectors.
"""
function run_scenario_repeats(label::String; kwargs...)
    all_ufp  = Float64[]
    all_pm25 = Float64[]
    for (rep, seed) in enumerate(SCENARIO_SEEDS)
        @printf("    Rep %d/%d (seed=%d)...", rep, N_REPEATS, seed)
        flush(stdout)
        t0 = time()
        result = run_moving_observer(; seed=seed, kwargs...)
        append!(all_ufp, result.ufp)
        append!(all_pm25, result.pm25)
        @printf(" done (%.1fs, %d samples)\n", time()-t0, length(result.ufp))
        flush(stdout)
        GC.gc()
    end
    return (ufp=all_ufp, pm25=all_pm25)
end

println("\nCommon setup loaded successfully.")
