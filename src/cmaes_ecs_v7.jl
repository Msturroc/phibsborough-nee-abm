# BIPOP-CMA-ES v7: Re-optimize using fast ECS v2 model
# Changes from v6:
#   - Uses phibsborough_ecs_v2.jl (ECS v2 model, ~7s/seed vs ~28s/seed)
#   - Tracks :observer instead of :e_suv
#   - Seeds: [12345, 22345, 32345] (first 3 from common_setup_v2.jl)
#   - No two-phase pruning — all candidates evaluated on all seeds
#   - Loss: heavier R²(q) emphasis (40% R², 25% RMSE, 25% KDE, 10% RMSElog)
#   - Robustness penalty: penalise if worst per-seed R² < 0.85
#   - Starts from best_parameters_cmaes_v4.txt
# Retained from v6:
#   - BIPOP restarts (alternating large/small populations)
#   - Per-dimension sigma scaling from parameter ranges
#   - Best-ever tracking + stagnation detection
#   - Condition number + 5 stopping criteria
#   - Cell-level spatial correlation (Spearman ρ at Air View GPS locations)
#   - Time-averaged grids (fast mode) with warmup, ensemble-averaged across repeats
#   - Checkpoint after each improvement

using LinearAlgebra
using Random
using DelimitedFiles
using CSV
using DataFrames
using Statistics
using StatsBase
using Printf
const HAS_PLOTS = try using Plots; using StatsPlots; true catch; false end

param_file = get(ENV, "PARAM_FILE", joinpath(@__DIR__, "..", "data", "best_parameters_cmaes_v7.txt"))
println("Loading parameters from: $param_file")

const GRID_SIZE = parse(Int, get(ENV, "GRID_SIZE", "200"))
ENV["GRID_SIZE"] = string(GRID_SIZE)

println("="^70)
println("BIPOP-CMA-ES v7 - ECS v2 + R²-HEAVY LOSS (no spatial in loss) + 3 SEEDS")
println("Grid size: $(GRID_SIZE)x$(GRID_SIZE)")
println("Threads available: $(Threads.nthreads())")
println("All candidates evaluated on all seeds (no pruning)")
println("="^70)

include("phibsborough_ecs_v2.jl")

# ============================================================================
# CMA-ES OPTIMIZER (with condition number tracking)
# ============================================================================

# CMA-ES operates in [0,1] normalized space internally.
# ask() returns real-space candidates; tell!() accepts real-space candidates.
# This ensures all dimensions are treated equally regardless of scale.

mutable struct CMAES
    N::Int
    λ::Int
    μ::Int
    weights::Vector{Float64}
    μeff::Float64
    cc::Float64
    cs::Float64
    c1::Float64
    cmu::Float64
    damps::Float64
    chiN::Float64
    xmean::Vector{Float64}   # in [0,1] normalized space
    sigma::Float64
    pc::Vector{Float64}
    ps::Vector{Float64}
    C::Matrix{Float64}
    B::Matrix{Float64}
    D::Vector{Float64}
    eigeneval::Int
    lb::Vector{Float64}      # real-space lower bounds
    ub::Vector{Float64}      # real-space upper bounds
    counteval::Int
    generation::Int
    ranges::Vector{Float64}  # ub - lb (for denormalization)
end

# Normalize real-space x to [0,1]
_to_unit(x, lb, ranges) = (x .- lb) ./ ranges
# Denormalize [0,1] to real-space
_to_real(u, lb, ranges) = lb .+ u .* ranges

function CMAES(xstart::Vector{Float64}, sigma::Float64;
               lb::Vector{Float64}=fill(-Inf, length(xstart)),
               ub::Vector{Float64}=fill(Inf, length(xstart)),
               popsize::Int=0)
    N = length(xstart)
    λ = popsize > 0 ? popsize : 4 + floor(Int, 3 * log(N))
    μ = λ ÷ 2
    raw_weights = [log(λ/2 + 0.5) - log(i) for i in 1:μ]
    weights = raw_weights ./ sum(raw_weights)
    μeff = sum(weights)^2 / sum(weights.^2)
    cc = (4 + μeff/N) / (N + 4 + 2*μeff/N)
    cs = (μeff + 2) / (N + μeff + 5)
    c1 = 2 / ((N + 1.3)^2 + μeff)
    cmu = min(1 - c1, 2 * (μeff - 2 + 1/μeff) / ((N + 2)^2 + μeff))
    damps = 2 * μeff / λ + 0.3 + cs
    chiN = sqrt(N) * (1 - 1/(4*N) + 1/(21*N^2))
    ranges = ub .- lb
    ranges = [r > 0 && isfinite(r) ? r : 1.0 for r in ranges]
    xmean = _to_unit(xstart, lb, ranges)
    pc = zeros(N)
    ps = zeros(N)
    C = Matrix{Float64}(I, N, N)  # identity — all dims equally scaled in [0,1]
    B = Matrix{Float64}(I, N, N)
    D = ones(N)
    CMAES(N, λ, μ, weights, μeff, cc, cs, c1, cmu, damps, chiN,
          xmean, sigma, pc, ps, C, B, D, 0, lb, ub, 0, 0, ranges)
end

function update_eigensystem!(es::CMAES)
    es.C .= (es.C .+ es.C') ./ 2
    F = eigen(Symmetric(es.C))
    es.D .= sqrt.(max.(F.values, 0.0))
    es.B .= F.vectors
    es.eigeneval = es.counteval
end

# Returns real-space candidates
function ask(es::CMAES)::Vector{Vector{Float64}}
    if es.counteval - es.eigeneval > es.λ / (es.c1 + es.cmu) / es.N / 10
        update_eigensystem!(es)
    end
    candidates = Vector{Vector{Float64}}(undef, es.λ)
    for k in 1:es.λ
        z = randn(es.N)
        y = es.B * (es.D .* z)
        u = es.xmean .+ es.sigma .* y
        u .= clamp.(u, 0.0, 1.0)
        candidates[k] = _to_real(u, es.lb, es.ranges)
    end
    return candidates
end

function ask_with_injection(es::CMAES)::Vector{Vector{Float64}}
    if es.counteval - es.eigeneval > es.λ / (es.c1 + es.cmu) / es.N / 10
        update_eigensystem!(es)
    end
    candidates = Vector{Vector{Float64}}(undef, es.λ)
    candidates[1] = _to_real(clamp.(es.xmean, 0.0, 1.0), es.lb, es.ranges)
    for k in 2:es.λ
        z = randn(es.N)
        y = es.B * (es.D .* z)
        u = es.xmean .+ es.sigma .* y
        u .= clamp.(u, 0.0, 1.0)
        candidates[k] = _to_real(u, es.lb, es.ranges)
    end
    return candidates
end

# tell! receives real-space candidates, converts to unit space internally
function tell!(es::CMAES, candidates::Vector{Vector{Float64}}, fitvals::Vector{Float64})
    es.counteval += es.λ
    es.generation += 1
    idx = sortperm(fitvals)
    # Convert real-space candidates to unit space
    unit_cands = [_to_unit(c, es.lb, es.ranges) for c in candidates]
    xold = copy(es.xmean)
    es.xmean .= sum(es.weights[i] .* unit_cands[idx[i]] for i in 1:es.μ)
    y = (es.xmean .- xold) ./ es.sigma
    z = es.B' * y
    z ./= (es.D .+ 1e-20)
    Cinvsqrt_y = es.B * z
    csn = sqrt(es.cs * (2 - es.cs) * es.μeff)
    es.ps .= (1 - es.cs) .* es.ps .+ csn .* Cinvsqrt_y
    pslen = norm(es.ps)
    threshold = (1.4 + 2/(es.N + 1)) * es.chiN * sqrt(1 - (1 - es.cs)^(2 * es.counteval / es.λ))
    hsig = pslen < threshold ? 1.0 : 0.0
    ccn = sqrt(es.cc * (2 - es.cc) * es.μeff)
    es.pc .= (1 - es.cc) .* es.pc .+ hsig * ccn .* y
    c1a = es.c1 * (1 - (1 - hsig^2) * es.cc * (2 - es.cc))
    rank_mu = zeros(es.N, es.N)
    for i in 1:es.μ
        yi = (unit_cands[idx[i]] .- xold) ./ es.sigma
        rank_mu .+= es.weights[i] .* (yi * yi')
    end
    es.C .= (1 - c1a - es.cmu * sum(es.weights)) .* es.C
    es.C .+= es.c1 .* (es.pc * es.pc')
    es.C .+= es.cmu .* rank_mu
    es.sigma *= exp(min(1, (es.cs / es.damps) * (pslen / es.chiN - 1) / 2))
    return fitvals[idx[1]], candidates[idx[1]]  # return real-space best
end

function condition_number(es::CMAES)
    if any(es.D .== 0)
        return Inf
    end
    return (maximum(es.D) / minimum(es.D[es.D .> 0]))^2
end

function should_stop(es::CMAES; tol::Float64=1e-12, max_cond::Float64=1e14, stagnation_count::Int=0, stagnation_limit::Int=20)
    # 1. Numerical error in C
    if any(isnan.(es.C)) || any(isinf.(es.C))
        return true, "Numerical error in C"
    end
    # 2. Numerical error in sigma
    if isnan(es.sigma) || isinf(es.sigma)
        return true, "Numerical error in sigma"
    end
    # 3. Sigma too small
    if es.sigma < tol
        return true, "Sigma below tolerance ($tol)"
    end
    # 4. Condition number too large
    cond = condition_number(es)
    if cond > max_cond
        return true, @sprintf("Condition number too large (%.2e > %.2e)", cond, max_cond)
    end
    # 5. Stagnation
    if stagnation_count >= stagnation_limit
        return true, "Stagnation ($stagnation_count gens without improvement)"
    end
    return false, ""
end

# ============================================================================
# LOSS FUNCTION - v7: heavier R²(q) emphasis
# 40% R²(q) + 25% RMSE linear + 25% KDE shape + 10% RMSE log
# ============================================================================

# Gaussian KDE on a fixed evaluation grid — lightweight, no external deps
function _kde_values(data::Vector{Float64}, grid::AbstractVector{Float64}, bw::Float64)
    n = length(data)
    inv_bw = 1.0 / bw
    norm = inv_bw / (n * sqrt(2π))
    density = zeros(length(grid))
    @inbounds for j in eachindex(data)
        dj = data[j]
        for i in eachindex(grid)
            z = (grid[i] - dj) * inv_bw
            density[i] += exp(-0.5 * z * z)
        end
    end
    density .*= norm
    return density
end

# KDE overlap loss: 1 - ∫ min(f_sim, f_exp) dx.  Naturally bounded [0,1], scale-invariant.
function kde_shape_loss(sim::Vector{Float64}, exp_data::Vector{Float64})
    lo = min(minimum(sim), minimum(exp_data))
    hi = max(maximum(sim), maximum(exp_data))
    rng = hi - lo
    if rng ≤ 0; return 0.0; end
    # Silverman bandwidth on experimental data (reference)
    bw = 1.06 * std(exp_data) * length(exp_data)^(-0.2)
    bw = max(bw, rng * 0.005)  # floor to avoid degenerate bandwidth
    grid = range(lo - 3bw, hi + 3bw, length=256)
    dx = step(grid)
    f_sim = _kde_values(sim, grid, bw)
    f_exp = _kde_values(exp_data, grid, bw)
    # Normalise both to integrate to 1
    f_sim ./= (sum(f_sim) * dx + 1e-30)
    f_exp ./= (sum(f_exp) * dx + 1e-30)
    # Overlap coefficient: ∫ min(f_sim, f_exp) dx ∈ [0,1]
    overlap = sum(min.(f_sim, f_exp)) * dx
    return 1.0 - clamp(overlap, 0.0, 1.0)
end

# Log-space KDE overlap: equalises main peak vs shoulder importance
function kde_shape_loss_log(sim::Vector{Float64}, exp_data::Vector{Float64})
    min_val = max(minimum(exp_data), 1.0)
    log_sim = log10.(max.(sim, min_val))
    log_exp = log10.(max.(exp_data, min_val))
    return kde_shape_loss(log_sim, log_exp)
end

function quantile_loss(sim_data, exp_data)
    clean_sim = Float64.(filter(!isnan, Float64.(sim_data)))
    clean_exp = Float64.(filter(!isnan, Float64.(exp_data)))

    if length(clean_sim) < 10 || length(clean_exp) < 10
        return 1.0, "NO_DATA", 0.0
    end

    exp_range = maximum(clean_exp) - minimum(clean_exp)
    if exp_range == 0
        return 1.0, "ZERO_RANGE", 0.0
    end

    quantiles = 0.01:0.01:0.99
    # Ramp up from q=0.50 to catch the right-hand shoulder
    weights = [q >= 0.95 ? 3.0 : (q >= 0.90 ? 2.5 : (q >= 0.80 ? 2.0 : (q >= 0.50 ? 1.5 : 1.0))) for q in quantiles]
    weights ./= mean(weights)

    q_sim = quantile(clean_sim, quantiles)
    q_exp = quantile(clean_exp, quantiles)
    # Asymmetric penalty: underestimating costs 3× more — force approach from above
    asym = [q_sim[i] < q_exp[i] ? 3.0 : 1.0 for (i, q) in enumerate(quantiles)]
    rmse_linear = sqrt(sum(weights .* asym .* (q_sim .- q_exp).^2) / sum(weights .* asym)) / exp_range

    min_val = max(minimum(clean_exp), 1.0)
    log_sim = log10.(max.(clean_sim, min_val))
    log_exp = log10.(max.(clean_exp, min_val))

    q_log_sim = quantile(log_sim, quantiles)
    q_log_exp = quantile(log_exp, quantiles)
    log_range = maximum(log_exp) - minimum(log_exp)
    rmse_log = log_range > 0 ? sqrt(sum(weights .* (q_log_sim .- q_log_exp).^2) / sum(weights)) / log_range : 0.0

    r2 = cor(q_exp, q_sim)^2

    # KDE shape penalty — directly catches bimodality
    shape = kde_shape_loss(clean_sim, clean_exp)
    # Log-space KDE — equalises main peak vs right-hand shoulder
    shape_log = kde_shape_loss_log(clean_sim, clean_exp)

    # v7 weights: 35% R², 20% RMSE linear, 15% KDE shape, 20% KDE shape(log), 10% RMSE log
    total_loss = 0.35 * (1.0 - r2) + 0.20 * min(rmse_linear, 1.0) +
                 0.15 * shape + 0.20 * shape_log + 0.10 * min(rmse_log, 1.0)

    status = @sprintf("RMSE=%.3f RMSElog=%.3f R²=%.3f shp=%.3f shpL=%.3f", rmse_linear, rmse_log, r2, shape, shape_log)
    return total_loss, status, r2
end

# ============================================================================
# LOAD EXPERIMENTAL DATA (200m × 200m box centered on junction)
# ============================================================================
println("\nLoading experimental data...")
lat_min, lat_max = 53.359899, 53.361701
lon_min, lon_max = -6.27421, -6.27119
full_airview_data = CSV.read(joinpath(@__DIR__, "..", "AirView_DublinCity_Measurements_ugm3.csv"), DataFrame)
phibs_data = filter(row -> (lat_min <= row.latitude <= lat_max) && (lon_min <= row.longitude <= lon_max), full_airview_data)
google_car_pm25_mass = Float64.(dropmissing(phibs_data, :PM25_ugm3).PM25_ugm3)
phibs_ufp_data = dropmissing(phibs_data, [:PMch1_perL, :PMch2_perL])
google_car_ufp_count = Float64.(phibs_ufp_data.PMch1_perL + phibs_ufp_data.PMch2_perL)
const BASELINE_UFP = minimum(google_car_ufp_count)
const BASELINE_PM25 = minimum(google_car_pm25_mass)
println("  UFP: n=$(length(google_car_ufp_count)), baseline=$BASELINE_UFP")
println("  PM2.5: n=$(length(google_car_pm25_mass)), baseline=$BASELINE_PM25")

# ============================================================================
# SPATIAL GRID: Bin Air View GPS data into 200×200 grid cells
# ============================================================================
println("\nBinning Air View data into $(GRID_SIZE)×$(GRID_SIZE) grid...")

const LAT_EDGES = range(lat_min, lat_max, length=GRID_SIZE+1)
const LON_EDGES = range(lon_min, lon_max, length=GRID_SIZE+1)

function bin_experimental_to_grid()
    # UFP grid
    ufp_rows = dropmissing(phibs_data, [:PMch1_perL, :PMch2_perL])
    ufp_sum = zeros(GRID_SIZE, GRID_SIZE)
    ufp_count = zeros(Int, GRID_SIZE, GRID_SIZE)
    for row in eachrow(ufp_rows)
        lat_idx = searchsortedlast(collect(LAT_EDGES), row.latitude)
        lon_idx = searchsortedlast(collect(LON_EDGES), row.longitude)
        if 1 <= lat_idx <= GRID_SIZE && 1 <= lon_idx <= GRID_SIZE
            ufp_sum[lat_idx, lon_idx] += Float64(row.PMch1_perL + row.PMch2_perL)
            ufp_count[lat_idx, lon_idx] += 1
        end
    end

    # PM2.5 grid
    pm25_rows = dropmissing(phibs_data, :PM25_ugm3)
    pm25_sum = zeros(GRID_SIZE, GRID_SIZE)
    pm25_count = zeros(Int, GRID_SIZE, GRID_SIZE)
    for row in eachrow(pm25_rows)
        lat_idx = searchsortedlast(collect(LAT_EDGES), row.latitude)
        lon_idx = searchsortedlast(collect(LON_EDGES), row.longitude)
        if 1 <= lat_idx <= GRID_SIZE && 1 <= lon_idx <= GRID_SIZE
            pm25_sum[lat_idx, lon_idx] += Float64(row.PM25_ugm3)
            pm25_count[lat_idx, lon_idx] += 1
        end
    end

    # Compute mean grids (NaN where no observations)
    ufp_grid = similar(ufp_sum)
    pm25_grid = similar(pm25_sum)
    for i in eachindex(ufp_sum)
        ufp_grid[i] = ufp_count[i] > 0 ? ufp_sum[i] / ufp_count[i] : NaN
    end
    for i in eachindex(pm25_sum)
        pm25_grid[i] = pm25_count[i] > 0 ? pm25_sum[i] / pm25_count[i] : NaN
    end

    return ufp_grid, pm25_grid, ufp_count, pm25_count
end

const EXP_UFP_GRID, EXP_PM25_GRID, COUNT_UFP_GRID, COUNT_PM25_GRID = bin_experimental_to_grid()

n_ufp_cells = sum(COUNT_UFP_GRID .> 0)
n_pm25_cells = sum(COUNT_PM25_GRID .> 0)
n_ufp_cells_3 = sum(COUNT_UFP_GRID .>= 3)
n_pm25_cells_3 = sum(COUNT_PM25_GRID .>= 3)
println("  UFP: $n_ufp_cells cells with data ($n_ufp_cells_3 with ≥3 obs)")
println("  PM2.5: $n_pm25_cells cells with data ($n_pm25_cells_3 with ≥3 obs)")

# ============================================================================
# SPATIAL LOSS HELPERS (radial profile correlation)
# ============================================================================
# Radial profile: bin cells by distance from junction centre, average within
# each ring, then correlate the decay curves.  This captures whether emissions
# fall off with distance from the junction — much more robust than cell-level.

const JUNCTION_CENTRE = (100, 100)
const DIST_GRID_XY = [sqrt((i - JUNCTION_CENTRE[1])^2 + (j - JUNCTION_CENTRE[2])^2)
                       for i in 1:GRID_SIZE, j in 1:GRID_SIZE]
const DIST_GRID_YX = DIST_GRID_XY'   # [lat,lon] orientation for experimental data
const RADIAL_BIN_EDGES = 0.0:5.0:100.0
const N_RADIAL_BINS = length(RADIAL_BIN_EDGES) - 1

function radial_profile(value_grid, dist_grid, count_grid; min_obs::Int=1)
    means = fill(NaN, N_RADIAL_BINS)
    for bi in 1:N_RADIAL_BINS
        r_lo, r_hi = RADIAL_BIN_EDGES[bi], RADIAL_BIN_EDGES[bi+1]
        mask = (dist_grid .>= r_lo) .& (dist_grid .< r_hi) .& (count_grid .>= min_obs)
        vals = filter(isfinite, value_grid[mask])
        if length(vals) >= 3
            means[bi] = mean(vals)
        end
    end
    return means
end

function radial_spatial_loss(model_grid::Matrix{Float64}, exp_grid::Matrix{Float64},
                              count_grid::Matrix{Int})
    # Model grid is [x,y]; use DIST_GRID_XY.  Exp grid is [lat,lon]=[y,x]; use DIST_GRID_YX.
    model_has_data = ones(Int, GRID_SIZE, GRID_SIZE)
    model_radial = radial_profile(model_grid, DIST_GRID_XY, model_has_data)
    exp_radial   = radial_profile(exp_grid, DIST_GRID_YX, count_grid)

    valid = isfinite.(model_radial) .& isfinite.(exp_radial)
    n = count(valid)
    if n < 3
        return 0.5, 0.0  # neutral if insufficient data
    end
    ρ = corspearman(model_radial[valid], exp_radial[valid])
    if isnan(ρ)
        return 0.5, 0.0
    end
    return (1.0 - ρ) / 2.0, ρ  # loss ∈ [0,1], raw ρ for logging
end

# ============================================================================
# BOUNDS
# ============================================================================
const LOWER_BOUNDS = Float64[log10(8.0), log10(0.5), log10(0.001), log10(0.01), log10(0.3), log10(0.3), log10(0.5), log10(0.3), log10(10.0), log10(1.0), log10(15.0), 1.0, 1.0, -2000.0, -0.5]
const UPPER_BOUNDS = Float64[log10(17.5), log10(5.0), log10(1e7), log10(500.0), log10(5.0), log10(200.0), log10(5.0), log10(6.0), log10(20.0), log10(10.0), log10(90.0), 6.0, 6.0, 2000.0, 2.0]

# ============================================================================
# EVALUATION FUNCTION
# All candidates evaluated on all seeds (no two-phase pruning)
# ============================================================================
const SEEDS = [12345, 22345, 32345]
const N_SEEDS = length(SEEDS)
const N_THREADS = Threads.nthreads()

# Single (candidate, seed) evaluation — the unit of parallelism
function evaluate_one_run(log10_params::Vector{Float64}, seed::Int)
    clamped = [clamp(log10_params[i], LOWER_BOUNDS[i], UPPER_BOUNDS[i]) for i in 1:length(log10_params)]
    params = 10 .^ clamped[1:11]

    dt = calculate_adaptive_dt(params[1], speed_variability=params[2])
    n_steps = round(Int, 1100.0 / dt)
    sample_interval = max(1, round(Int, 1.0 / dt))
    warmup_steps = round(Int, 100.0 / dt)

    run_sim_args = Dict{Symbol, Any}(
        :dims => (GRID_SIZE, GRID_SIZE), :mean_speed => params[1],
        :speed_variability => params[2], :brake_decay_rate => params[5],
        :tyre_decay_rate => params[6], :max_sight_distance => round(Int, params[9]),
        :planning_distance => params[10], :random_seed => seed,
        :green_duration_s => params[11], :amber_duration_s => 3.0,
        :red_duration_s => params[11] + 3.0,
        :brake_dispersion_radius => clamp(round(Int, clamped[12]), 1, 6),
        :tyre_dispersion_radius => clamp(round(Int, clamped[13]), 1, 6),
        :steps => n_steps,
        :data_sample_interval => sample_interval,
        :tracked_vehicle_type => :observer,
        :fast_mode => true,
        :grid_warmup_steps => warmup_steps)

    _, observer_data = run_simulation(; run_sim_args...)
    all_data = observer_data["all_tracked_vehicles"]
    brake_raw, tyre_raw = all_data[:brake_emissions][1], all_data[:tyre_emissions][1]
    step_times = [(i-1)*dt*sample_interval for i in 1:length(brake_raw)]

    brake_sampled, tyre_sampled = Float64[], Float64[]
    for t in 100.0:1.0:1100.0
        idx_t = clamp(searchsortedlast(step_times, t), 1, length(step_times))
        push!(brake_sampled, brake_raw[idx_t]); push!(tyre_sampled, tyre_raw[idx_t])
    end

    sim_ufp = (BASELINE_UFP + clamped[14]) .+ params[3] .* (brake_sampled .^ params[7])
    sim_pm25 = (BASELINE_PM25 + clamped[15]) .+ params[4] .* (tyre_sampled .^ params[8])

    # Radial spatial loss from this run's time-averaged grids
    avg_brake_grid = observer_data["avg_brake_grid"]
    avg_tyre_grid = observer_data["avg_tyre_grid"]
    ufp_grid = (BASELINE_UFP + clamped[14]) .+ params[3] .* (avg_brake_grid .^ params[7])
    pm25_grid = (BASELINE_PM25 + clamped[15]) .+ params[4] .* (avg_tyre_grid .^ params[8])

    ufp_spatial, ufp_rho = radial_spatial_loss(ufp_grid, EXP_UFP_GRID, COUNT_UFP_GRID)
    pm25_spatial, pm25_rho = radial_spatial_loss(pm25_grid, EXP_PM25_GRID, COUNT_PM25_GRID)

    # Return raw arrays (pooled in aggregate_results) + spatial scalars
    return (sim_ufp, sim_pm25, ufp_spatial, pm25_spatial)
end

# Aggregate run results: GEOMEAN of per-seed losses (no seed can hide behind others)
function aggregate_results(run_results::Vector)
    # Per-seed distributional loss (geometric mean — one bad seed tanks the score)
    seed_ufp_losses = Float64[]
    seed_pm25_losses = Float64[]
    per_seed_r2s = Float64[]
    seed_ufp_statuses = String[]
    seed_pm25_statuses = String[]
    for r in run_results
        ul, us, r2u = quantile_loss(r[1], google_car_ufp_count)
        pl, ps, r2p = quantile_loss(r[2], google_car_pm25_mass)
        push!(seed_ufp_losses, ul)
        push!(seed_pm25_losses, pl)
        push!(per_seed_r2s, r2u)
        push!(per_seed_r2s, r2p)
        push!(seed_ufp_statuses, us)
        push!(seed_pm25_statuses, ps)
    end

    # Geometric mean of per-seed losses (add ε to avoid log(0))
    ε = 1e-10
    ufp_loss  = exp(mean(log.(seed_ufp_losses .+ ε)))
    pm25_loss = exp(mean(log.(seed_pm25_losses .+ ε)))
    ufp_status  = seed_ufp_statuses[argmin(seed_ufp_losses)]   # show best seed's breakdown
    pm25_status = seed_pm25_statuses[argmin(seed_pm25_losses)]

    # Spatial loss (still per-seed average — grids are single-seed)
    ufp_spatial  = mean(r[3] for r in run_results)
    pm25_spatial = mean(r[4] for r in run_results)

    worst_r2 = minimum(per_seed_r2s)

    # Weights: 50/50 UFP/PM2.5 — geomean already penalises bad seeds
    # Radial spatial ρ still computed for monitoring but NOT included in loss
    total_loss = 0.50 * ufp_loss + 0.50 * pm25_loss

    spatial_status = @sprintf("ρ_UFP=%.2f ρ_PM25=%.2f worst_R²=%.3f",
                              1.0 - 2.0 * ufp_spatial, 1.0 - 2.0 * pm25_spatial,
                              worst_r2)
    return total_loss, ufp_loss, pm25_loss, ufp_status, pm25_status, ufp_spatial, pm25_spatial, spatial_status, per_seed_r2s
end

# Convenience: evaluate a single candidate sequentially (for initial eval)
function evaluate_single(log10_params::Vector{Float64}, idx::Int)
    run_results = [evaluate_one_run(log10_params, seed) for seed in SEEDS]
    return aggregate_results(run_results)
end

# Full evaluation: ALL candidates on ALL seeds — no pruning (ECS v2 is fast enough)
function evaluate_population_parallel(candidates::Vector{Vector{Float64}})
    n = length(candidates)

    # Build flat task list: n candidates × N_SEEDS seeds
    n_tasks = n * N_SEEDS
    task_results = Vector{Tuple{Vector{Float64},Vector{Float64},Float64,Float64}}(undef, n_tasks)

    Threads.@threads :dynamic for k in 1:n_tasks
        ci = (k - 1) ÷ N_SEEDS + 1  # candidate index
        si = (k - 1) % N_SEEDS + 1   # seed index
        task_results[k] = evaluate_one_run(candidates[ci], SEEDS[si])
    end

    # Aggregate per candidate
    results = Vector{Tuple{Float64,Float64,Float64,String,String,Float64,Float64,String,Vector{Float64}}}(undef, n)
    for i in 1:n
        start_k = (i - 1) * N_SEEDS + 1
        end_k = i * N_SEEDS
        all_runs = [task_results[k] for k in start_k:end_k]
        results[i] = aggregate_results(all_runs)
    end
    return results
end

# ============================================================================
# VALIDATION PLOT
# ============================================================================
function generate_validation_plot(log10_params::Vector{Float64}, loss::Float64, gen::Int;
                                   restart::Int=0, per_seed_r2s::Vector{Float64}=Float64[])
    if !HAS_PLOTS
        println("  (skipping validation plot — Plots not available)")
        return
    end
    clamped = [clamp(log10_params[i], LOWER_BOUNDS[i], UPPER_BOUNDS[i]) for i in 1:length(log10_params)]
    params = 10 .^ clamped[1:11]
    dt = calculate_adaptive_dt(params[1], speed_variability=params[2])

    # Run all 3 seeds and pool — shows the ACTUAL distribution the scenario figures will display
    sim_ufp = Float64[]
    sim_pm25 = Float64[]
    for seed in SEEDS
        run_sim_args = Dict{Symbol, Any}(
            :dims => (GRID_SIZE, GRID_SIZE), :mean_speed => params[1],
            :speed_variability => params[2], :brake_decay_rate => params[5],
            :tyre_decay_rate => params[6], :max_sight_distance => round(Int, params[9]),
            :planning_distance => params[10], :random_seed => seed,
            :green_duration_s => params[11], :amber_duration_s => 3.0,
            :red_duration_s => params[11] + 3.0,
            :brake_dispersion_radius => clamp(round(Int, clamped[12]), 1, 6),
            :tyre_dispersion_radius => clamp(round(Int, clamped[13]), 1, 6),
            :steps => round(Int, 1100.0 / dt),
            :data_sample_interval => max(1, round(Int, 1.0 / dt)),
            :tracked_vehicle_type => :observer)

        _, observer_data = run_simulation(; run_sim_args...)
        all_data = observer_data["all_tracked_vehicles"]
        brake_raw, tyre_raw = all_data[:brake_emissions][1], all_data[:tyre_emissions][1]
        step_times = [(i-1)*dt*run_sim_args[:data_sample_interval] for i in 1:length(brake_raw)]

        for t in 100.0:1.0:1100.0
            idx_t = clamp(searchsortedlast(step_times, t), 1, length(step_times))
            push!(sim_ufp, (BASELINE_UFP + clamped[14]) + params[3] * (brake_raw[idx_t] ^ params[7]))
            push!(sim_pm25, (BASELINE_PM25 + clamped[15]) + params[4] * (tyre_raw[idx_t] ^ params[8]))
        end
    end

    pm25_bw = 1.06 * std(google_car_pm25_mass) * length(google_car_pm25_mass)^(-0.2)
    ufp_bw = 1.06 * std(google_car_ufp_count) * length(google_car_ufp_count)^(-0.2)

    # Row 1: Density plots
    p1 = density(google_car_pm25_mass, label="Experimental", lw=2.5, color=:blue, bandwidth=pm25_bw)
    density!(p1, sim_pm25, label="Model", lw=2.5, color=:coral, ls=:dash, bandwidth=pm25_bw)
    title!(p1, "PM2.5 Distribution")
    xlabel!(p1, "PM2.5 (μg/m³)")
    ylabel!(p1, "Density")

    p2 = density(google_car_ufp_count, label="Experimental", lw=2.5, color=:blue, bandwidth=ufp_bw)
    density!(p2, sim_ufp, label="Model", lw=2.5, color=:coral, ls=:dash, bandwidth=ufp_bw)
    title!(p2, "UFP Distribution")
    xlabel!(p2, "UFP (particles/L)")
    ylabel!(p2, "Density")

    # Row 2: Q-Q plots
    n_quantiles = 100
    probs = range(0.01, 0.99, length=n_quantiles)
    exp_pm25_quantiles = quantile(google_car_pm25_mass, probs)
    sim_pm25_quantiles = quantile(sim_pm25, probs)
    pm25_qq_min = min(minimum(exp_pm25_quantiles), minimum(sim_pm25_quantiles))
    pm25_qq_max = max(maximum(exp_pm25_quantiles), maximum(sim_pm25_quantiles))

    p3 = scatter(exp_pm25_quantiles, sim_pm25_quantiles, label="Quantiles",
                 color=:steelblue, alpha=0.7, ms=4)
    plot!(p3, [pm25_qq_min, pm25_qq_max], [pm25_qq_min, pm25_qq_max],
          label="y=x", color=:red, lw=2, ls=:dash)
    title!(p3, "PM2.5 Q-Q Plot")
    xlabel!(p3, "Experimental Quantiles (μg/m³)")
    ylabel!(p3, "Model Quantiles (μg/m³)")

    exp_ufp_quantiles = quantile(google_car_ufp_count, probs)
    sim_ufp_quantiles = quantile(sim_ufp, probs)
    ufp_qq_min = min(minimum(exp_ufp_quantiles), minimum(sim_ufp_quantiles))
    ufp_qq_max = max(maximum(exp_ufp_quantiles), maximum(sim_ufp_quantiles))

    p4 = scatter(exp_ufp_quantiles, sim_ufp_quantiles, label="Quantiles",
                 color=:steelblue, alpha=0.7, ms=4)
    plot!(p4, [ufp_qq_min, ufp_qq_max], [ufp_qq_min, ufp_qq_max],
          label="y=x", color=:red, lw=2, ls=:dash)
    title!(p4, "UFP Q-Q Plot")
    xlabel!(p4, "Experimental UFP (particles/L)")
    ylabel!(p4, "Model UFP (particles/L)")

    restart_str = restart > 0 ? " R$restart" : ""
    # Add per-seed R² info to title if available
    r2_str = ""
    if length(per_seed_r2s) >= 2 * N_SEEDS
        r2_parts = String[]
        for s in 1:N_SEEDS
            push!(r2_parts, @sprintf("S%d: UFP=%.2f PM25=%.2f", s, per_seed_r2s[2*s-1], per_seed_r2s[2*s]))
        end
        r2_str = " | " * join(r2_parts, " ")
    end
    combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900),
                    plot_title=@sprintf("Gen %d%s | Loss=%.4f%s", gen, restart_str, loss, r2_str),
                    margin=5Plots.mm)
    savefig(combined, joinpath(@__DIR__, "..", "current_best_validation_v7.png"))
    println("  → Saved validation plot to current_best_validation_v7.png")
end

# ============================================================================
# BIPOP-CMA-ES MAIN LOOP
# ============================================================================

# Load starting point
if isfile(param_file)
    println("Loading starting point from: $param_file")
    log10_params = vec(readdlm(param_file))
    if length(log10_params) != length(LOWER_BOUNDS)
        println("Warning: Parameter file has $(length(log10_params)) params, expected $(length(LOWER_BOUNDS))")
        println("Starting from middle of bounds instead")
        log10_params = (LOWER_BOUNDS .+ UPPER_BOUNDS) ./ 2.0
    end
else
    log10_params = (LOWER_BOUNDS .+ UPPER_BOUNDS) ./ 2.0
    println("Starting from middle of bounds (unbiased)")
end

# Configuration
N_dim = length(log10_params)
# 10 pop × 3 seeds = 30 tasks per generation → maps to ~30 threads
default_pop = try parse(Int, get(ENV, "POP_SIZE", "10")) catch; 10 end
max_total_evals = try parse(Int, get(ENV, "MAX_EVALS", "1000")) catch; 1000 end
ini_sigma = try parse(Float64, get(ENV, "INI_SIGMA", "0.15")) catch; 0.15 end
stagnation_limit = 20

println("\nBIPOP Configuration:")
println("  Dimension: $N_dim")
println("  Default population: $default_pop")
println("  Seeds: $N_SEEDS ($(join(SEEDS, ", ")))")
println("  Tasks per generation: $(default_pop) × $N_SEEDS = $(default_pop * N_SEEDS)")
println("  Threads available: $N_THREADS")
println("  Total budget: $max_total_evals evaluations")
println("  Initial σ: $ini_sigma")
println("  Stagnation limit: $stagnation_limit generations")

# Global best tracking
global best_loss = Inf
global best_params = copy(log10_params)
global total_evals = 0

# Initial evaluation
println("\nEvaluating starting point...")
init_result = evaluate_single(log10_params, 0)
init_loss = init_result[1]
init_r2s = init_result[9]
@printf("Start: %.4f (UFP=%.4f [%s], PM25=%.4f [%s] | %s)\n",
        init_loss, init_result[2], init_result[4], init_result[3], init_result[5], init_result[8])
# Print per-seed R²
for s in 1:N_SEEDS
    @printf("  Seed %d: R²_UFP=%.4f R²_PM25=%.4f\n", SEEDS[s], init_r2s[2*s-1], init_r2s[2*s])
end
total_evals += 1

if init_loss < best_loss
    global best_loss = init_loss
    global best_params = copy(log10_params)
    writedlm(joinpath(@__DIR__, "..", "data", "best_parameters_cmaes_v7.txt"), best_params)
    generate_validation_plot(best_params, best_loss, 0; per_seed_r2s=init_r2s)
end

# BIPOP restart loop
max_restarts = 9
large_evals_used = 0  # Budget tracking for BIPOP regime selection

println("\n" * "="^70)
println("Starting BIPOP-CMA-ES optimization...")
println("="^70)

for restart in 0:max_restarts
    global total_evals, best_loss, best_params

    if total_evals >= max_total_evals
        println("\nBudget exhausted ($total_evals/$max_total_evals evals)")
        break
    end

    # BIPOP regime selection
    if restart == 0
        # First run: default population with warm start
        pop_size = default_pop
        σ0 = ini_sigma
        x0 = copy(best_params)
        regime = "DEFAULT"
    elseif restart % 2 == 1
        # Odd restarts: LARGE population (2× default)
        pop_size = 2 * default_pop
        σ0 = ini_sigma * 1.5  # Broader exploration
        # Start from best-ever with perturbation
        x0 = best_params .+ 0.1 .* (UPPER_BOUNDS .- LOWER_BOUNDS) .* randn(N_dim)
        x0 .= clamp.(x0, LOWER_BOUNDS, UPPER_BOUNDS)
        regime = "LARGE"
    else
        # Even restarts: SMALL population (default / 2, min 4)
        pop_size = max(4, default_pop ÷ 2)
        σ0 = ini_sigma * 0.5  # Focused local search
        x0 = copy(best_params)  # Start from best-ever
        regime = "SMALL"
    end

    remaining_evals = max_total_evals - total_evals
    max_iters_this_run = remaining_evals ÷ pop_size

    if max_iters_this_run < 3
        println("\nInsufficient budget for restart $restart (only $remaining_evals evals left)")
        break
    end

    println("\n" * "-"^50)
    @printf("RESTART %d [%s] | pop=%d | σ=%.3f | budget=%d evals | global_best=%.4f\n",
            restart, regime, pop_size, σ0, remaining_evals, best_loss)
    println("-"^50)

    es = CMAES(x0, σ0; lb=LOWER_BOUNDS, ub=UPPER_BOUNDS, popsize=pop_size)
    stagnation_count = 0
    run_best_loss = Inf

    for iter in 1:max_iters_this_run
        if total_evals >= max_total_evals
            break
        end

        candidates = (iter == 1 && restart == 0) ? ask_with_injection(es) : ask(es)
        results = evaluate_population_parallel(candidates)
        fitvals = [r[1] for r in results]
        total_evals += pop_size

        iter_best, iter_best_x = tell!(es, candidates, fitvals)

        # Track stagnation within this run
        if iter_best < run_best_loss - 1e-6
            run_best_loss = iter_best
            stagnation_count = 0
        else
            stagnation_count += 1
        end

        # Update global best
        improved = false
        best_idx = argmin(fitvals)
        best_r2s = results[best_idx][9]
        if iter_best < best_loss
            best_loss = iter_best
            best_params = copy(iter_best_x)
            writedlm(joinpath(@__DIR__, "..", "data", "best_parameters_cmaes_v7.txt"), best_params)
            generate_validation_plot(best_params, best_loss, iter; restart=restart, per_seed_r2s=best_r2s)
            improved = true
        end

        ufp_loss, pm25_loss = results[best_idx][2], results[best_idx][3]
        ufp_s, pm25_s = results[best_idx][4], results[best_idx][5]
        spatial_s = results[best_idx][8]

        # Print per-seed R² for monitoring
        r2_parts = String[]
        for s in 1:N_SEEDS
            push!(r2_parts, @sprintf("S%d:U%.2f/P%.2f", s, best_r2s[2*s-1], best_r2s[2*s]))
        end
        r2_str = join(r2_parts, " ")

        marker = improved ? "★" : " "
        cond = condition_number(es)
        @printf("%s R%d G%3d: best=%.4f (global=%.4f) | UFP=%.3f PM25=%.3f | %s | %s | σ=%.4f cond=%.1e | %d/%d evals\n",
                marker, restart, iter, iter_best, best_loss,
                ufp_loss, pm25_loss, spatial_s, r2_str, es.sigma, cond,
                total_evals, max_total_evals)
        flush(stdout)

        # 5-point stopping criteria
        stop, reason = should_stop(es; stagnation_count=stagnation_count, stagnation_limit=stagnation_limit)
        if stop
            println("  → Run stopped: $reason")
            break
        end
    end

    @printf("  Run %d finished: run_best=%.4f, global_best=%.4f, evals_so_far=%d\n",
            restart, run_best_loss, best_loss, total_evals)
end

println("\n" * "="^70)
println("BIPOP-CMA-ES v7 COMPLETE")
println("="^70)
@printf("  Best loss: %.4f\n", best_loss)
println("  Total evaluations: $total_evals")
println("  Saved to: best_parameters_cmaes_v7.txt")
println("="^70)
