#!/usr/bin/env julia
"""
Global Sensitivity Analysis (GSA) - Passenger-Preserving Fleet Composition
Updated for phibsborough_ecs.jl (ECS port) and best_parameters_cmaes_v4.txt

POLICY-FOCUSED: Measures ABSOLUTE DIFFERENCE in NEE exposure vs calibrated baseline.
  Δ emissions = scenario - baseline
  - Positive = MORE emissions than baseline (worse, e.g. heavier E-SUVs)
  - Negative = LESS emissions than baseline (better, e.g. modal shift to buses)

Sobol indices reveal which parameters cause the most variance in outcomes.

POLICY-MOTIVATED RANGES: Absolute ranges reflecting achievable urban transport
policies (speed limits, signal timing, junction geometry) rather than ±20% local
perturbation, so Sobol indices reflect true global sensitivity.
Uses MOVING OBSERVER (tracked e-SUV) with 400s timing (100s warmup + 300s data).

Parameters with clear physical interpretations (11 total):

Fleet Composition (6) - passenger-preserving mode shares:
  1. ICE Car share
  2. Electric Car share
  3. ICE SUV share
  4. Electric SUV share
  5. Bus share
  6. Microcar share

Driving/Traffic (5) - policy-motivated absolute ranges:
  7. Mean speed: 30-50 km/h (8.33-13.89 m/s) - speed limit policies
  8. Speed variability: 0.25-3.0 m/s - smooth (CAV) to aggressive stop-start
  9. Green duration: 15-45 s - pedestrian-priority to vehicle-priority cycles
  10. Sight distance: 5-17 cells - junction geometry improvements
  11. Planning distance: calibrated ± 20% - not a direct policy lever

Key policy questions this GSA answers:
- Is electrification effective, or do heavier EVs increase NEEs?
- How much can modal shift (buses, microcars) reduce NEEs?
- How important is traffic flow vs fleet composition?

Uses PCE (Polynomial Chaos Expansion) for Sobol index computation.
"""

using Distributed

# Add worker processes
n_workers = try parse(Int, get(ENV, "GSA_WORKERS", "24")) catch; 24 end
if nprocs() < n_workers + 1
    addprocs(n_workers - nprocs() + 1)
end
println("Workers: $(nprocs() - 1)")

@everywhere begin
    using Random
    using Statistics
    using Distributions
    using DelimitedFiles
    using CSV
    using DataFrames
end

using QuasiMonteCarlo
using SurrogatesPolyChaos
using Plots
using Printf
using StatsBase

# Output directories
const GSA_FIGURES_DIR = joinpath(@__DIR__, "..", "Figures")
const GSA_DATA_DIR = joinpath(@__DIR__, "..", "data")

# Grid size (match calibration)
const GRID_SIZE = parse(Int, get(ENV, "GRID_SIZE", "200"))
@everywhere const GRID_SIZE = $GRID_SIZE
@everywhere ENV["GRID_SIZE"] = string(GRID_SIZE)

# Include model on all workers (ECS port: bit-perfect, ~20x faster)
println("Loading model on all workers...")
const _MODEL_PATH = joinpath(@__DIR__, "phibsborough_ecs_v2.jl")
@everywhere include($_MODEL_PATH)
@everywhere const calculate_adaptive_dt = calculate_adaptive_dt_ecs
@everywhere const REGEN_EFF = REGEN_EFF_ECS

# Load calibrated parameters (for fixed/baseline values)
const PARAM_FILE = get(ENV, "PARAM_FILE", joinpath(@__DIR__, "..", "data", "best_parameters_cmaes_v7_excellent_contender.txt"))
println("Loading calibrated parameters from: $PARAM_FILE")
const BEST_LOG10_PARAMS = vec(readdlm(PARAM_FILE))

# Broadcast to workers
@everywhere const BEST_LOG10_PARAMS = $BEST_LOG10_PARAMS
@everywhere const BASELINE_SEEDS = [92345, 22345, 62345]

# Parameter bounds from cmaes_parallel_v4_revised.jl (with realistic speed_variability)
# Note: speed_variability bounds (param 2) updated to realistic values
# OLD: log10(0.00001) to log10(0.1) = 0.00001-0.1 m/s (CV <1% - unrealistic)
# NEW: log10(0.5) to log10(5.0) = 0.5-5.0 m/s (CV 4-40% - realistic traffic)
@everywhere const CMAES_LOWER = Float64[log10(8.0), log10(0.5), log10(0.001), log10(0.01), log10(0.3), log10(0.3), log10(0.5), log10(0.3), log10(10.0), log10(1.0), log10(15.0), 1.0, 1.0, -2000.0, -0.5]
@everywhere const CMAES_UPPER = Float64[log10(17.5), log10(5.0), log10(1e7), log10(500.0), log10(5.0), log10(200.0), log10(5.0), log10(6.0), log10(20.0), log10(10.0), log10(90.0), 6.0, 6.0, 2000.0, 2.0]

# Load experimental data for validation metrics
println("Loading experimental data...")
const _AIRVIEW_PATH = joinpath(@__DIR__, "..", "AirView_DublinCity_Measurements_ugm3.csv")
@everywhere begin
    lat_min, lat_max = 53.359899, 53.361701
    lon_min, lon_max = -6.27421, -6.27119
    full_airview = CSV.read($_AIRVIEW_PATH, DataFrame)
    phibs_data = filter(row -> (lat_min <= row.latitude <= lat_max) && (lon_min <= row.longitude <= lon_max), full_airview)

    google_car_pm25 = Float64.(dropmissing(phibs_data, :PM25_ugm3).PM25_ugm3)
    phibs_ufp = dropmissing(phibs_data, [:PMch1_perL, :PMch2_perL])
    google_car_ufp = Float64.(phibs_ufp.PMch1_perL + phibs_ufp.PMch2_perL)

    const BASELINE_UFP = minimum(google_car_ufp)
    const BASELINE_PM25 = minimum(google_car_pm25)
end
println("  UFP: n=$(length(google_car_ufp)), PM2.5: n=$(length(google_car_pm25))")

# ============================================================================
# FLEET COMPOSITION - PASSENGER PRESERVING
# ============================================================================

@everywhere begin
    # Baseline fleet (per direction) - matches calibration scenario
    const BASE_CARS  = 10
    const BASE_ECARS = 1
    const BASE_SUVS  = 12
    const BASE_ESUVS = 1
    const BASE_BUSES = 1
    const BASE_MICRO = 0  # Microcars for GSA scenarios

    # Occupancies (passengers per vehicle)
    const OCC_CAR   = 1.27
    const OCC_ECAR  = 1.27
    const OCC_SUV   = 1.27
    const OCC_ESUV  = 1.27
    const OCC_BUS   = 22.5
    const OCC_MICRO = 1.10

    # Total baseline passengers per direction
    const BASELINE_PASSENGERS = BASE_CARS * OCC_CAR + BASE_ECARS * OCC_ECAR +
                                BASE_SUVS * OCC_SUV + BASE_ESUVS * OCC_ESUV +
                                BASE_BUSES * OCC_BUS

    # Baseline passenger shares
    const BASE_PSHARE_CAR   = (BASE_CARS * OCC_CAR) / BASELINE_PASSENGERS
    const BASE_PSHARE_ECAR  = (BASE_ECARS * OCC_ECAR) / BASELINE_PASSENGERS
    const BASE_PSHARE_SUV   = (BASE_SUVS * OCC_SUV) / BASELINE_PASSENGERS
    const BASE_PSHARE_ESUV  = (BASE_ESUVS * OCC_ESUV) / BASELINE_PASSENGERS
    const BASE_PSHARE_BUS   = (BASE_BUSES * OCC_BUS) / BASELINE_PASSENGERS
    const BASE_PSHARE_MICRO = 0.01  # Small baseline for microcars
end

@everywhere function softmax(y::AbstractVector{<:Real})
    ym = maximum(y)
    ex = exp.(y .- ym)
    ex ./ sum(ex)
end

@everywhere function allocate_vehicles_passenger_preserving(
    shares::NTuple{6,Float64},  # (car, ecar, suv, esuv, bus, micro)
    Ptot::Float64
)
    s_c, s_ec, s_s, s_es, s_b, s_mc = shares
    occs = (OCC_CAR, OCC_ECAR, OCC_SUV, OCC_ESUV, OCC_BUS, OCC_MICRO)

    # Real-valued vehicle counts from passenger shares
    cr = [(s * Ptot) / o for (s, o) in zip(shares, occs)]

    # Start from floors
    n = [floor(Int, c) for c in cr]

    # Observer is now a separate VT_OBSERVER vehicle — no minimum e-SUV constraint needed

    # Greedy adjustment to match Ptot
    passengers(nv) = sum(nv[i] * occs[i] for i in 1:6)
    res = Ptot - passengers(n)

    for _ in 1:100
        abs(res) < 0.5 && break
        if res > 0
            # Add vehicle that best reduces residual
            best_i = argmin([abs(res - o) for o in occs])
            n[best_i] += 1
        else
            # Remove vehicle (if > 0)
            candidates = [(abs(res + occs[i]), i) for i in 1:6 if n[i] > 0]
            isempty(candidates) && break
            _, best_i = minimum(candidates)
            n[best_i] -= 1
        end
        res = Ptot - passengers(n)
    end

    return (n[1], n[2], n[3], n[4], n[5], n[6])
end

# ============================================================================
# GSA PARAMETER MAPPING
# ============================================================================

# Parameter names for plots
const PARAM_NAMES = [
    "Cars", "E-Cars", "SUVs", "E-SUVs", "Buses", "Microcars",
    "Mean Speed", "Speed Variability", "Green Duration",
    "Sight Distance", "Planning Distance"
]
const N_PARAMS = 11

@everywhere function gsa_params_to_simulation(p::Vector{Float64})
    """
    Map GSA parameters [0,1]^11 to simulation parameters.
    Independent vehicle counts + policy-motivated driving ranges.

    p[1]    - Cars: 0-20
    p[2]    - E-Cars: 0-10
    p[3]    - SUVs: 0-20
    p[4]    - E-SUVs: 0-10
    p[5]    - Buses: 0-3
    p[6]    - Microcars: 0-10
    p[7]    - Mean speed: 30-50 km/h (8.33-13.89 m/s)
    p[8]    - Speed variability: 0.25-3.0 m/s
    p[9]    - Green duration: 15-45 s
    p[10]   - Sight distance: 5-17 cells
    p[11]   - Planning distance: calibrated ± 20%
    """

    # Get calibrated values
    clamped = [clamp(BEST_LOG10_PARAMS[i], CMAES_LOWER[i], CMAES_UPPER[i]) for i in 1:length(BEST_LOG10_PARAMS)]
    calib = 10 .^ clamped[1:11]

    # p ∈ [0,1] maps linearly to [lo, hi]
    lerp(x, lo, hi) = lo + (hi - lo) * x

    # --- Fleet composition: independent vehicle counts ---
    n_c  = round(Int, lerp(p[1], 0, 15))   # Cars: 0-15 (3 cells each)
    n_ec = round(Int, lerp(p[2], 0, 5))    # E-Cars: 0-5 (3 cells each)
    n_s  = round(Int, lerp(p[3], 0, 15))   # SUVs: 0-15 (4 cells each)
    n_es = round(Int, lerp(p[4], 0, 5))    # E-SUVs: 0-5 (4 cells each)
    n_b  = round(Int, lerp(p[5], 0, 3))    # Buses: 0-3 (7 cells each)
    n_mc = round(Int, lerp(p[6], 0, 5))    # Microcars: 0-5 (2 cells each)

    # --- Driving/traffic parameters: policy-motivated absolute ranges ---
    mean_speed    = lerp(p[7], 30.0/3.6, 50.0/3.6)  # 30-50 km/h → 8.33-13.89 m/s
    speed_var     = lerp(p[8], 0.25, 3.0)              # 0.25-3.0 m/s
    green_dur     = lerp(p[9], 15.0, 45.0)             # 15-45 s signal cycles
    sight_dist    = lerp(p[10], 5.0, 17.0)             # 5-17 cells
    planning_dist = calib[10] * (0.8 + 0.4 * p[11])   # keep ±20%

    # --- Emission physics: fixed at calibrated values ---
    brake_decay = calib[5]
    tyre_decay = calib[6]
    brake_disp = clamp(round(Int, clamped[12]), 1, 6)

    # Bus occupancy: fixed at baseline
    bus_occ = OCC_BUS

    return (
        n_cars = n_c, n_ecars = n_ec, n_suvs = n_s, n_esuvs = n_es, n_buses = n_b, n_micros = n_mc,
        mean_speed = mean_speed, speed_var = speed_var, green_dur = green_dur,
        sight_dist = sight_dist, planning_dist = planning_dist,
        brake_decay = brake_decay, tyre_decay = tyre_decay, brake_disp = brake_disp,
        bus_occupancy = bus_occ
    )
end

# ============================================================================
# SIMULATION AND OUTPUT METRICS
# ============================================================================

@everywhere function run_gsa_simulation(p::Vector{Float64}, baseline_ufp::Float64, baseline_pm25::Float64; seed::Int=12345, sample_id::Int=0)
    """
    Run simulation with GSA parameters and return output metrics.
    Runs 4 seeds per sample and averages results for noise reduction.
    Returns tuple: (delta_total, delta_ufp, delta_pm25) - differences from baseline.
    """
    try
        sim_params = gsa_params_to_simulation(p)

        # Check fleet fits on grid before running simulation
        # Each direction has 40% of grid edge × 2 lanes for spawning
        # Use 80% of theoretical capacity as safety margin for random placement
        spawn_cells_per_direction = round(Int, GRID_SIZE * 0.4) * 2
        cells_needed = 3 * (sim_params.n_cars + sim_params.n_ecars) +
                       4 * (sim_params.n_suvs + sim_params.n_esuvs) +
                       7 * sim_params.n_buses +
                       2 * sim_params.n_micros
        if cells_needed > spawn_cells_per_direction * 0.8
            @warn "Sample $sample_id: Skipping - fleet too large for grid ($cells_needed cells needed, $(round(Int, spawn_cells_per_direction * 0.8)) available per direction)"
            return (NaN, NaN, NaN)
        end

        # Use calibrated values for non-GSA parameters
        clamped = [clamp(BEST_LOG10_PARAMS[i], CMAES_LOWER[i], CMAES_UPPER[i]) for i in 1:length(BEST_LOG10_PARAMS)]
        calib = 10 .^ clamped[1:11]

        # Build simulation args - 400s total (100s warmup + 300s data)
        dt = calculate_adaptive_dt(sim_params.mean_speed, speed_variability=sim_params.speed_var)
        total_time = 400.0
        warmup_time = 100.0
        n_steps = round(Int, total_time / dt)

        # Skip if too many steps (would take too long)
        MAX_STEPS = 8000  # ~10s simulation time limit
        if n_steps > MAX_STEPS
            @warn "Sample $sample_id: Skipping - too many steps ($n_steps > $MAX_STEPS). speed=$(round(sim_params.mean_speed, digits=2)), dt=$(round(dt, digits=4))"
            return (NaN, NaN, NaN)
        end

        # Log parameters for this simulation
        @info "Sample $sample_id: speed=$(round(sim_params.mean_speed, digits=1))m/s, dt=$(round(dt, digits=3))s, steps=$n_steps, fleet=[$(sim_params.n_cars),$(sim_params.n_ecars),$(sim_params.n_suvs),$(sim_params.n_esuvs),$(sim_params.n_buses),$(sim_params.n_micros)]"

        run_args = Dict{Symbol, Any}(
            :dims => (GRID_SIZE, GRID_SIZE),
            :mean_speed => sim_params.mean_speed,
            :speed_variability => sim_params.speed_var,
            :brake_decay_rate => sim_params.brake_decay,
            :tyre_decay_rate => sim_params.tyre_decay,
            :max_sight_distance => round(Int, sim_params.sight_dist),  # GSA parameter
            :planning_distance => sim_params.planning_dist,             # GSA parameter
            :green_duration_s => sim_params.green_dur,
            :amber_duration_s => 3.0,
            :red_duration_s => sim_params.green_dur + 3.0,
            :brake_dispersion_radius => sim_params.brake_disp,
            :tyre_dispersion_radius => clamp(round(Int, clamped[13]), 1, 6),  # From calibration
            :steps => n_steps,
            :data_sample_interval => max(1, round(Int, 1.0 / dt)),
            :tracked_vehicle_type => :observer,
            :bus_base_weight => 12000.0,
            :bus_occupancy => sim_params.bus_occupancy,
            :fast_mode => true,
            :grid_warmup_steps => round(Int, warmup_time / dt),
            # Fleet composition (same for all directions)
            :n_cars_left => sim_params.n_cars, :n_cars_right => sim_params.n_cars,
            :n_cars_up => sim_params.n_cars, :n_cars_down => sim_params.n_cars,
            :n_e_cars_left => sim_params.n_ecars, :n_e_cars_right => sim_params.n_ecars,
            :n_e_cars_up => sim_params.n_ecars, :n_e_cars_down => sim_params.n_ecars,
            :n_suvs_left => sim_params.n_suvs, :n_suvs_right => sim_params.n_suvs,
            :n_suvs_up => sim_params.n_suvs, :n_suvs_down => sim_params.n_suvs,
            :n_e_suvs_left => sim_params.n_esuvs, :n_e_suvs_right => sim_params.n_esuvs,
            :n_e_suvs_up => sim_params.n_esuvs, :n_e_suvs_down => sim_params.n_esuvs,
            :n_buses_left => sim_params.n_buses, :n_buses_right => sim_params.n_buses,
            :n_buses_up => sim_params.n_buses, :n_buses_down => sim_params.n_buses,
            :n_smart_cars_left => sim_params.n_micros, :n_smart_cars_right => sim_params.n_micros,
            :n_smart_cars_up => sim_params.n_micros, :n_smart_cars_down => sim_params.n_micros
        )

        # Run 3 seeds and average results for noise reduction (same seeds as baseline)
        seeds = BASELINE_SEEDS
        delta_totals, delta_ufps, delta_pm25s = Float64[], Float64[], Float64[]

        for s in seeds
            run_args[:random_seed] = s

            _, observer_data = run_simulation(; run_args...)

            # Extract emissions from tracked vehicle (moving observer)
            all_data = observer_data["all_tracked_vehicles"]
            brake_raw = all_data[:brake_emissions][1]
            tyre_raw = all_data[:tyre_emissions][1]

            # Sample at 1-second intervals after warmup
            step_times = [(i-1)*dt*run_args[:data_sample_interval] for i in 1:length(brake_raw)]
            brake_sampled, tyre_sampled = Float64[], Float64[]
            for t in warmup_time:1.0:total_time
                idx_t = clamp(searchsortedlast(step_times, t), 1, length(step_times))
                push!(brake_sampled, brake_raw[idx_t])
                push!(tyre_sampled, tyre_raw[idx_t])
            end

            # Convert to concentrations using calibrated scaling
            sim_ufp = (BASELINE_UFP + clamped[14]) .+ calib[3] .* (brake_sampled .^ calib[7])
            sim_pm25 = (BASELINE_PM25 + clamped[15]) .+ calib[4] .* (tyre_sampled .^ calib[8])

            # Output metrics: DIFFERENCE from baseline (positive = worse, negative = better)
            ufp_mean = mean(sim_ufp)
            pm25_mean = mean(sim_pm25)

            # Differences from baseline (passed as arguments)
            delta_ufp = ufp_mean - baseline_ufp
            delta_pm25 = pm25_mean - baseline_pm25
            # Combined: scale PM2.5 to similar magnitude as UFP for combined metric
            delta_total = delta_ufp + delta_pm25 * 1000

            push!(delta_totals, delta_total)
            push!(delta_ufps, delta_ufp)
            push!(delta_pm25s, delta_pm25)
        end

        return (mean(delta_totals), mean(delta_ufps), mean(delta_pm25s))
    catch e
        @warn "Simulation failed" exception=(e, catch_backtrace())
        return (NaN, NaN, NaN)
    end
end

# ============================================================================
# BASELINE SIMULATION
# ============================================================================

function run_baseline_simulation(; seed::Int=12345, n_runs::Int=length(BASELINE_SEEDS))
    """
    Run baseline simulation with calibrated parameters.
    Returns mean UFP and PM2.5 exposure (averaged over multiple runs for stability).
    """
    println("Computing baseline emissions ($(n_runs) runs, seeds=$(BASELINE_SEEDS))...")
    flush(stdout)

    clamped = [clamp(BEST_LOG10_PARAMS[i], CMAES_LOWER[i], CMAES_UPPER[i]) for i in 1:length(BEST_LOG10_PARAMS)]
    calib = 10 .^ clamped[1:11]

    ufp_runs = Float64[]
    pm25_runs = Float64[]

    dt = calculate_adaptive_dt(calib[1], speed_variability=calib[2])
    total_time = 400.0
    warmup_time = 100.0

    for run in 1:n_runs
        run_args = Dict{Symbol, Any}(
            :dims => (GRID_SIZE, GRID_SIZE),
            :mean_speed => calib[1],
            :speed_variability => calib[2],
            :brake_decay_rate => calib[5],
            :tyre_decay_rate => calib[6],
            :max_sight_distance => round(Int, calib[9]),
            :planning_distance => calib[10],
            :random_seed => BASELINE_SEEDS[run],
            :green_duration_s => calib[11],
            :amber_duration_s => 3.0,
            :red_duration_s => calib[11] + 3.0,
            :brake_dispersion_radius => clamp(round(Int, clamped[12]), 1, 6),
            :tyre_dispersion_radius => clamp(round(Int, clamped[13]), 1, 6),
            :steps => round(Int, total_time / dt),
            :data_sample_interval => max(1, round(Int, 1.0 / dt)),
            :tracked_vehicle_type => :observer,
            :bus_base_weight => 12000.0,
            :bus_occupancy => OCC_BUS,
            :fast_mode => true,
            :grid_warmup_steps => round(Int, warmup_time / dt),
            # Baseline fleet (per direction)
            :n_cars_left => BASE_CARS, :n_cars_right => BASE_CARS,
            :n_cars_up => BASE_CARS, :n_cars_down => BASE_CARS,
            :n_e_cars_left => BASE_ECARS, :n_e_cars_right => BASE_ECARS,
            :n_e_cars_up => BASE_ECARS, :n_e_cars_down => BASE_ECARS,
            :n_suvs_left => BASE_SUVS, :n_suvs_right => BASE_SUVS,
            :n_suvs_up => BASE_SUVS, :n_suvs_down => BASE_SUVS,
            :n_e_suvs_left => BASE_ESUVS, :n_e_suvs_right => BASE_ESUVS,
            :n_e_suvs_up => BASE_ESUVS, :n_e_suvs_down => BASE_ESUVS,
            :n_buses_left => BASE_BUSES, :n_buses_right => BASE_BUSES,
            :n_buses_up => BASE_BUSES, :n_buses_down => BASE_BUSES,
            :n_smart_cars_left => BASE_MICRO, :n_smart_cars_right => BASE_MICRO,
            :n_smart_cars_up => BASE_MICRO, :n_smart_cars_down => BASE_MICRO
        )

        _, observer_data = run_simulation(; run_args...)

        all_data = observer_data["all_tracked_vehicles"]
        brake_raw = all_data[:brake_emissions][1]
        tyre_raw = all_data[:tyre_emissions][1]

        step_times = [(i-1)*dt*run_args[:data_sample_interval] for i in 1:length(brake_raw)]
        brake_sampled, tyre_sampled = Float64[], Float64[]
        for t in warmup_time:1.0:total_time
            idx_t = clamp(searchsortedlast(step_times, t), 1, length(step_times))
            push!(brake_sampled, brake_raw[idx_t])
            push!(tyre_sampled, tyre_raw[idx_t])
        end

        sim_ufp = (BASELINE_UFP + clamped[14]) .+ calib[3] .* (brake_sampled .^ calib[7])
        sim_pm25 = (BASELINE_PM25 + clamped[15]) .+ calib[4] .* (tyre_sampled .^ calib[8])

        push!(ufp_runs, mean(sim_ufp))
        push!(pm25_runs, mean(sim_pm25))
        println("    Baseline run $run/$n_runs complete")
        flush(stdout)
    end

    baseline_ufp = mean(ufp_runs)
    baseline_pm25 = mean(pm25_runs)

    println("  Baseline UFP:   $(round(baseline_ufp, digits=1)) #/L")
    println("  Baseline PM2.5: $(round(baseline_pm25, digits=2)) μg/m³")
    flush(stdout)

    return (ufp=baseline_ufp, pm25=baseline_pm25)
end

# Global baseline values (computed once at startup)
const BASELINE_EMISSIONS = Ref((ufp=0.0, pm25=0.0))

# ============================================================================
# SOBOL INDEX COMPUTATION VIA PCE
# ============================================================================

function compute_sobol_indices(pce)
    """Compute ST (total), S1 (first-order), S2 (interaction) from PCE coefficients."""
    coeffs = pce.coeff
    multiidx = pce.orthopolys.ind
    d = size(multiidx, 2)

    varY = sum(coeffs[2:end].^2)
    if varY < 1e-12
        return (zeros(d), zeros(d), zeros(d, d))
    end

    ST = zeros(d)
    S1 = zeros(d)
    S2 = zeros(d, d)

    for k in 2:length(coeffs)
        c2 = coeffs[k]^2
        active = findall(multiidx[k, :] .> 0)

        # Total indices
        for i in active
            ST[i] += c2
        end

        # First-order
        if length(active) == 1
            S1[active[1]] += c2
        # Second-order interactions
        elseif length(active) == 2
            i, j = active[1], active[2]
            S2[i, j] += c2
            S2[j, i] = S2[i, j]
        end
    end

    return (ST ./ varY, S1 ./ varY, S2 ./ varY)
end

# ============================================================================
# MAIN GSA ROUTINE
# ============================================================================

function run_gsa(baseline_ufp::Float64, baseline_pm25::Float64; n_samples::Int=500, n_repeats::Int=3, output_idx::Int=1, start_repeat::Int=1, suffix::String="", pce_degree::Int=2)
    """
    Run GSA with PCE-based Sobol indices.

    Args:
        baseline_ufp: Baseline UFP concentration for computing differences
        baseline_pm25: Baseline PM2.5 concentration for computing differences
        n_samples: Number of Sobol sequence samples per repeat
        n_repeats: Number of independent repeats for confidence intervals
        output_idx: 1=Δ total, 2=Δ UFP, 3=Δ PM2.5
        start_repeat: Repeat to start from (>1 to resume a previous run)
    """
    output_names = ["Δ Total NEE", "Δ UFP", "Δ PM2.5"]
    output_name = output_names[output_idx]

    println("\n" * "="^70)
    println("GSA: $(output_name) | Samples=$n_samples | Repeats=$n_repeats")
    println("="^70)

    d = N_PARAMS
    lb, ub = zeros(d), ones(d)

    # Storage for results across repeats (load previous if resuming)
    if start_repeat > 1
        println("\nResuming from repeat $start_repeat (loading repeats 1-$(start_repeat-1))...")
        flush(stdout)
        ST_all, S1_all, S2_all, n_loaded = load_raw_repeats(output_name, n_repeats; suffix=suffix)
        if n_loaded < start_repeat - 1
            error("Expected $(start_repeat-1) completed repeats but found $n_loaded")
        end
    else
        ST_all = zeros(n_repeats, d)
        S1_all = zeros(n_repeats, d)
        S2_all = [zeros(d, d) for _ in 1:n_repeats]
    end

    for rep in start_repeat:n_repeats
        println("\nRepeat $rep/$n_repeats...")
        flush(stdout)

        # Generate Sobol sequence with scrambling
        Random.seed!(42 + rep)
        sampler = SobolSample(QuasiMonteCarlo.Shift())
        X = QuasiMonteCarlo.sample(d, n_samples, sampler)  # n_samples × d

        # Evaluate simulations in batches with progress reporting
        println("  Evaluating $n_samples simulations...")
        flush(stdout)
        Y = Vector{Float64}(undef, n_samples)

        batch_size = min(n_samples, nworkers() * 2)  # 2 samples per worker per batch
        n_batches = ceil(Int, n_samples / batch_size)
        rep_start = time()

        for batch in 1:n_batches
            batch_start = (batch - 1) * batch_size + 1
            batch_end = min(batch * batch_size, n_samples)
            batch_indices = batch_start:batch_end

            batch_t = time()
            batch_results = pmap(batch_indices) do i
                xi = Vector(X[i, :])
                out = run_gsa_simulation(xi, baseline_ufp, baseline_pm25; seed=12345 + i, sample_id=i)
                out[output_idx]
            end
            Y[batch_indices] .= batch_results
            batch_elapsed = time() - batch_t

            # Progress report
            completed = batch_end
            elapsed = time() - rep_start
            rate = completed / elapsed
            eta = (n_samples - completed) / rate
            @printf("    Batch %d/%d: samples %d-%d done (%.1fs). Progress: %d/%d (%.0f%%). ETA: %.0fs\n",
                batch, n_batches, batch_start, batch_end, batch_elapsed,
                completed, n_samples, 100*completed/n_samples, eta)
            flush(stdout)
        end

        rep_elapsed = time() - rep_start
        @printf("  Repeat %d complete in %.1fs (%.2fs/sample)\n", rep, rep_elapsed, rep_elapsed/n_samples)
        flush(stdout)

        # Filter NaNs
        valid_idx = findall(!isnan, Y)
        n_valid = length(valid_idx)
        println("  Valid samples: $n_valid / $n_samples")
        flush(stdout)

        if n_valid < 50
            @warn "Too few valid samples, skipping repeat $rep"
            continue
        end

        Xg = [Vector(X[i, :]) for i in valid_idx]
        Yg = Y[valid_idx]

        # Build PCE surrogate
        println("  Building PCE surrogate (degree $pce_degree)...")
        flush(stdout)
        orthos = SurrogatesPolyChaos.MultiOrthoPoly(
            [SurrogatesPolyChaos.GaussOrthoPoly(pce_degree) for _ in 1:d], pce_degree
        )
        pce = SurrogatesPolyChaos.PolynomialChaosSurrogate(Xg, Yg, lb, ub; orthopolys=orthos)

        # Compute Sobol indices
        ST, S1, S2 = compute_sobol_indices(pce)
        ST_all[rep, :] = ST
        S1_all[rep, :] = S1
        S2_all[rep] = S2

        @printf("  Top 3 ST: %s=%.3f, %s=%.3f, %s=%.3f\n",
            PARAM_NAMES[sortperm(ST, rev=true)[1]], sort(ST, rev=true)[1],
            PARAM_NAMES[sortperm(ST, rev=true)[2]], sort(ST, rev=true)[2],
            PARAM_NAMES[sortperm(ST, rev=true)[3]], sort(ST, rev=true)[3])
        flush(stdout)

        # Save intermediate results after each repeat
        ST_mean_so_far = vec(mean(ST_all[1:rep, :], dims=1))
        ST_std_so_far = rep > 1 ? vec(std(ST_all[1:rep, :], dims=1)) : fill(NaN, d)
        S1_mean_so_far = vec(mean(S1_all[1:rep, :], dims=1))
        S1_std_so_far = rep > 1 ? vec(std(S1_all[1:rep, :], dims=1)) : fill(NaN, d)
        S2_mean_so_far = reduce(.+, S2_all[1:rep]) ./ rep

        intermediate = (ST=ST_mean_so_far, ST_std=ST_std_so_far,
                        S1=S1_mean_so_far, S1_std=S1_std_so_far,
                        S2=S2_mean_so_far, output_name=output_name)

        plot_sobol_results(intermediate; suffix=suffix)
        save_sobol_csv(intermediate; suffix=suffix)
        save_raw_repeats(ST_all, S1_all, S2_all, rep, output_name; suffix=suffix)
        println("  Intermediate results saved (repeats 1-$rep)")
        flush(stdout)
    end

    # Aggregate results
    ST_mean = vec(mean(ST_all, dims=1))
    ST_std = vec(std(ST_all, dims=1))
    S1_mean = vec(mean(S1_all, dims=1))
    S1_std = vec(std(S1_all, dims=1))
    S2_mean = reduce(.+, S2_all) ./ n_repeats

    return (ST=ST_mean, ST_std=ST_std, S1=S1_mean, S1_std=S1_std, S2=S2_mean, output_name=output_name)
end

# ============================================================================
# PLOTTING
# ============================================================================

function plot_sobol_results(results; suffix::String="")
    mkpath(GSA_FIGURES_DIR)
    tag = replace(lowercase(results.output_name), " " => "_")

    # Sort by ST (descending) for clearer visualization
    sorted_idx = sortperm(results.ST, rev=true)
    sorted_names = PARAM_NAMES[sorted_idx]
    sorted_ST = results.ST[sorted_idx]
    sorted_ST_std = results.ST_std[sorted_idx]
    sorted_S1 = results.S1[sorted_idx]
    sorted_S1_std = results.S1_std[sorted_idx]

    # Handle NaN std (single repeat case) - replace with zeros for plotting
    plot_ST_std = replace(sorted_ST_std, NaN => 0.0)
    plot_S1_std = replace(sorted_S1_std, NaN => 0.0)

    # Total Sobol indices (ST) - scatter plot with error bars (sorted by importance)
    p_st = scatter(1:N_PARAMS, sorted_ST, yerr=plot_ST_std,
        xlabel="Parameters", ylabel="Total Sobol (ST)",
        xticks=(1:N_PARAMS, sorted_names), xrotation=45,
        legend=false, color=:steelblue,
        markersize=6, markerstrokewidth=1,
        size=(900, 450), bottom_margin=15Plots.mm, left_margin=10Plots.mm,
        grid=true, gridalpha=0.3)
    savefig(p_st, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_st_sorted$(suffix).pdf"))
    savefig(p_st, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_st_sorted$(suffix).png"))

    # Also save unsorted version (original parameter order)
    p_st_unsorted = scatter(1:N_PARAMS, results.ST, yerr=replace(results.ST_std, NaN => 0.0),
        xlabel="Parameters", ylabel="Total Sobol (ST)",
        xticks=(1:N_PARAMS, PARAM_NAMES), xrotation=45,
        legend=false, color=:steelblue,
        markersize=6, markerstrokewidth=1,
        size=(900, 450), bottom_margin=15Plots.mm, left_margin=10Plots.mm,
        grid=true, gridalpha=0.3)
    savefig(p_st_unsorted, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_st$(suffix).pdf"))
    savefig(p_st_unsorted, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_st$(suffix).png"))

    # First-order Sobol indices (S1) - scatter plot with error bars (sorted by ST importance)
    p_s1 = scatter(1:N_PARAMS, sorted_S1, yerr=plot_S1_std,
        xlabel="Parameters", ylabel="First-Order Sobol (S1)",
        xticks=(1:N_PARAMS, sorted_names), xrotation=45,
        legend=false, color=:coral,
        markersize=6, markerstrokewidth=1,
        size=(900, 450), bottom_margin=15Plots.mm, left_margin=10Plots.mm,
        grid=true, gridalpha=0.3)
    savefig(p_s1, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_s1$(suffix).pdf"))
    savefig(p_s1, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_s1$(suffix).png"))

    # Interaction heatmap (S2) - lower triangle only
    S2_lower = copy(results.S2)
    for i in 1:N_PARAMS, j in i:N_PARAMS
        S2_lower[i, j] = NaN
    end

    p_s2 = heatmap(PARAM_NAMES, PARAM_NAMES, S2_lower,
        title="Second-Order Interactions (S2) - $(results.output_name)",
        c=:viridis, clims=(0, maximum(filter(!isnan, S2_lower))),
        size=(900, 800), xrotation=45, yflip=true,
        bottom_margin=15Plots.mm, left_margin=15Plots.mm)
    savefig(p_s2, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_s2$(suffix).pdf"))
    savefig(p_s2, joinpath(GSA_FIGURES_DIR, "gsa_$(tag)_s2$(suffix).png"))

    println("Saved plots to $(GSA_FIGURES_DIR)/gsa_$(tag)_*.pdf/png")
end

function save_raw_repeats(ST_all, S1_all, S2_all, n_completed::Int, output_name::String; suffix::String="")
    """Save raw per-repeat arrays so GSA can be resumed later."""
    mkpath(GSA_DATA_DIR)
    tag = replace(lowercase(output_name), " " => "_")
    fname = joinpath(GSA_DATA_DIR, "gsa_$(tag)_raw_repeats$(suffix).csv")

    open(fname, "w") do io
        println(io, "n_completed=$n_completed")
        println(io, "n_params=$(size(ST_all, 2))")

        for rep in 1:n_completed
            println(io, "# repeat $rep ST")
            println(io, join(ST_all[rep, :], ","))
            println(io, "# repeat $rep S1")
            println(io, join(S1_all[rep, :], ","))
            println(io, "# repeat $rep S2")
            for i in 1:size(S2_all[rep], 1)
                println(io, join(S2_all[rep][i, :], ","))
            end
        end
    end
    println("  Raw repeat data saved to $fname")
end

function load_raw_repeats(output_name::String, n_repeats::Int; suffix::String="")
    """Load raw per-repeat arrays from a previous run."""
    tag = replace(lowercase(output_name), " " => "_")
    fname = joinpath(GSA_DATA_DIR, "gsa_$(tag)_raw_repeats$(suffix).csv")

    if !isfile(fname)
        error("Resume file not found: $fname")
    end

    lines = readlines(fname)
    n_completed = parse(Int, split(lines[1], "=")[2])
    d = parse(Int, split(lines[2], "=")[2])

    ST_all = zeros(n_repeats, d)
    S1_all = zeros(n_repeats, d)
    S2_all = [zeros(d, d) for _ in 1:n_repeats]

    idx = 3
    for rep in 1:n_completed
        idx += 1  # skip "# repeat N ST"
        ST_all[rep, :] = parse.(Float64, split(lines[idx], ","))
        idx += 1
        idx += 1  # skip "# repeat N S1"
        S1_all[rep, :] = parse.(Float64, split(lines[idx], ","))
        idx += 1
        idx += 1  # skip "# repeat N S2"
        for i in 1:d
            S2_all[rep][i, :] = parse.(Float64, split(lines[idx], ","))
            idx += 1
        end
    end

    println("  Loaded $n_completed completed repeats from $fname")
    return ST_all, S1_all, S2_all, n_completed
end

function save_sobol_csv(results; suffix::String="")
    mkpath(GSA_DATA_DIR)
    tag = replace(lowercase(results.output_name), " " => "_")

    # ST and S1
    open(joinpath(GSA_DATA_DIR, "gsa_$(tag)_indices$(suffix).csv"), "w") do io
        println(io, "parameter,ST_mean,ST_std,S1_mean,S1_std")
        for i in 1:N_PARAMS
            @printf(io, "%s,%.6f,%.6f,%.6f,%.6f\n",
                PARAM_NAMES[i], results.ST[i], results.ST_std[i], results.S1[i], results.S1_std[i])
        end
    end

    # S2 matrix
    open(joinpath(GSA_DATA_DIR, "gsa_$(tag)_s2_matrix$(suffix).csv"), "w") do io
        println(io, "," * join(PARAM_NAMES, ","))
        for i in 1:N_PARAMS
            print(io, PARAM_NAMES[i])
            for j in 1:N_PARAMS
                @printf(io, ",%.6f", results.S2[i, j])
            end
            println(io)
        end
    end

    println("Saved CSV to $(GSA_DATA_DIR)/gsa_$(tag)_*.csv")
end

# ============================================================================
# MAIN
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    n_samples = try parse(Int, get(ENV, "GSA_SAMPLES", "1000")) catch; 1000 end
    n_repeats = try parse(Int, get(ENV, "GSA_REPEATS", "5")) catch; 5 end
    output_idx = try parse(Int, get(ENV, "GSA_OUTPUT", "1")) catch; 1 end
    start_repeat = try parse(Int, get(ENV, "GSA_START_REPEAT", "1")) catch; 1 end
    suffix = get(ENV, "GSA_SUFFIX", "")
    pce_degree = try parse(Int, get(ENV, "GSA_PCE_DEGREE", "2")) catch; 2 end

    println("\n" * "="^70)
    println("PASSENGER-PRESERVING GLOBAL SENSITIVITY ANALYSIS")
    println("Model: phibsborough_ecs_v2.jl (ECS port + dedicated observer)")
    println("Parameters: $PARAM_FILE")
    println("Grid: $(GRID_SIZE)×$(GRID_SIZE)")
    println("Output: Δ from baseline (positive=worse, negative=better)")
    println("PCE degree: $pce_degree")
    println("="^70)

    # Compute baseline first
    start_time = time()
    BASELINE_EMISSIONS[] = run_baseline_simulation(n_runs=3)
    baseline_time = time() - start_time
    println("Baseline computed in $(round(baseline_time, digits=1))s")

    # Run GSA
    gsa_start = time()
    if start_repeat > 1
        println("Resuming from repeat $start_repeat/$n_repeats")
    end

    results = run_gsa(BASELINE_EMISSIONS[].ufp, BASELINE_EMISSIONS[].pm25;
                       n_samples=n_samples, n_repeats=n_repeats, output_idx=output_idx, start_repeat=start_repeat, suffix=suffix, pce_degree=pce_degree)
    gsa_time = time() - gsa_start

    println("\n" * "="^70)
    println("RESULTS SUMMARY - $(results.output_name)")
    println("="^70)

    # Sort by ST
    sorted_idx = sortperm(results.ST, rev=true)
    println("\nRanked by Total Sobol Index (ST):")
    for (rank, i) in enumerate(sorted_idx)
        @printf("  %2d. %-18s  ST=%.4f ± %.4f  S1=%.4f ± %.4f\n",
            rank, PARAM_NAMES[i], results.ST[i], results.ST_std[i], results.S1[i], results.S1_std[i])
    end

    plot_sobol_results(results; suffix=suffix)
    save_sobol_csv(results; suffix=suffix)

    # Timing summary
    total_time = baseline_time + gsa_time
    evals_per_hour = n_samples * n_repeats / (gsa_time / 3600)

    println("\n" * "="^70)
    println("GSA COMPLETE")
    println("="^70)
    println("\nTiming Summary:")
    println("  Baseline:    $(round(baseline_time, digits=1))s")
    println("  GSA:         $(round(gsa_time/60, digits=1)) min")
    println("  Total:       $(round(total_time/60, digits=1)) min")
    println("  Throughput:  $(round(Int, evals_per_hour)) evals/hour")
    println("  Per eval:    $(round(gsa_time/(n_samples*n_repeats), digits=2))s")
end
