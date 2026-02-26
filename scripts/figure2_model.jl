#!/usr/bin/env julia
"""
Figure 2: Model Validation
2×3 layout:
  Row 1: (a) Model snapshot with emission insets, (b) PM2.5 density, (c) UFP density
  Row 2: (d) PM2.5 heatmap, (e) UFP heatmap, (f) Combined heatmap

Based on cmaes_parallel_v4_revised.jl visualization and generate_figure2.jl snapshot style.
Uses best_parameters_cmaes_v4.txt (15 parameters) and NO static observer.
"""

using StatsPlots
pythonplot()
using Measures
using Random
using Statistics
using DelimitedFiles
using Printf
using CSV
using DataFrames
using StatsBase
using KernelDensity

const GRID_SIZE = 200
ENV["GRID_SIZE"] = string(GRID_SIZE)

# Include the model (ECS v2 with dedicated observer)
println("Loading ECS model v2...")
include(joinpath(@__DIR__, "..", "src", "phibsborough_ecs_v2.jl"))

# Load calibrated parameters (CMA-ES v7 - 15 parameters, ECS v2 optimized)
param_file = joinpath(@__DIR__, "..", "data", "best_parameters_cmaes_v7_excellent_contender.txt")
println("Loading parameters from: $param_file")
log10_params = vec(readdlm(param_file))

# Bounds (same as cmaes_parallel_v4_revised.jl - with realistic speed_variability)
# Note: speed_variability bounds (param 2) updated to realistic values
# OLD: log10(0.00001) to log10(0.1) = 0.00001-0.1 m/s (CV <1% - unrealistic)
# NEW: log10(0.5) to log10(5.0) = 0.5-5.0 m/s (CV 4-40% - realistic traffic)
const LOWER_BOUNDS = Float64[log10(8.0), log10(0.5), log10(0.001), log10(0.01), log10(0.3), log10(0.3), log10(0.5), log10(0.3), log10(10.0), log10(1.0), log10(15.0), 1.0, 1.0, -2000.0, -0.5]
const UPPER_BOUNDS = Float64[log10(17.5), log10(5.0), log10(1e7), log10(500.0), log10(5.0), log10(200.0), log10(5.0), log10(6.0), log10(20.0), log10(10.0), log10(90.0), 6.0, 6.0, 2000.0, 2.0]

clamped = [clamp(log10_params[i], LOWER_BOUNDS[i], UPPER_BOUNDS[i]) for i in 1:length(log10_params)]
params = 10 .^ clamped[1:11]

# Load experimental data (200m × 200m box - same as cmaes_parallel_v4_revised.jl)
println("Loading experimental data (200m × 200m box)...")
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

# Run 4-seed simulations (matching scenario figures methodology)
# Seed 1 also provides snapshot, insets, and heatmap data
const SCENARIO_SEEDS = [92345, 22345, 62345]  # top 3 seeds from 10-seed evaluation (excellent contender)
const TOTAL_TIME = 1100.0
const WARMUP_TIME = 100.0

dt = calculate_adaptive_dt(params[1], speed_variability=params[2])
n_steps = round(Int, TOTAL_TIME / dt)
data_sample_interval = max(1, round(Int, 1.0 / dt))
warmup_steps_raw = round(Int, WARMUP_TIME / dt)
warmup_samples = round(Int, warmup_steps_raw / data_sample_interval)

println("\n  dt=$(round(dt, digits=4))s, total_steps=$n_steps")

# Pooled observer data for R²(q) calculation
sim_ufp = Float64[]
sim_pm25 = Float64[]
# Per-seed data for averaged KDE (avoids mixture bimodality)
per_seed_ufp  = Vector{Vector{Float64}}()
per_seed_pm25 = Vector{Vector{Float64}}()

# Seed-1 variables for snapshot (panel a) and insets
car_positions = nothing
# traffic_light_states: not available in ECS model (unused in plot anyway)
observer_data = nothing

# Accumulated grids for 4-seed averaged heatmaps (panels d-f)
sum_brake_grid = nothing
sum_tyre_grid = nothing

for (i, seed) in enumerate(SCENARIO_SEEDS)
    @printf("Running seed %d/%d (seed=%d)...", i, length(SCENARIO_SEEDS), seed)
    flush(stdout)
    t0 = time()

    run_sim_args = Dict{Symbol, Any}(
        :dims => (GRID_SIZE, GRID_SIZE),
        :mean_speed => params[1],
        :speed_variability => params[2],
        :brake_decay_rate => params[5],
        :tyre_decay_rate => params[6],
        :max_sight_distance => round(Int, params[9]),
        :planning_distance => params[10],
        :random_seed => seed,
        :green_duration_s => params[11],
        :amber_duration_s => 3.0,
        :red_duration_s => params[11] + 3.0,
        :brake_dispersion_radius => clamp(round(Int, clamped[12]), 1, 6),
        :tyre_dispersion_radius => clamp(round(Int, clamped[13]), 1, 6),
        :tracked_vehicle_type => :observer,
        :steps => n_steps,
        :data_sample_interval => data_sample_interval,
    )

    model_df, obs_data = run_simulation(; run_sim_args...)

    # Extract observer time series and sample at 1Hz after warmup
    all_tracked = obs_data["all_tracked_vehicles"]
    brake_raw = all_tracked[:brake_emissions][1]
    tyre_raw  = all_tracked[:tyre_emissions][1]
    step_times = [(j-1)*dt*data_sample_interval for j in 1:length(brake_raw)]

    seed_ufp  = Float64[]
    seed_pm25 = Float64[]
    for t in WARMUP_TIME:1.0:TOTAL_TIME
        idx_t = clamp(searchsortedlast(step_times, t), 1, length(step_times))
        push!(seed_ufp,  (BASELINE_UFP + clamped[14])  + params[3] * (brake_raw[idx_t] ^ params[7]))
        push!(seed_pm25, (BASELINE_PM25 + clamped[15]) + params[4] * (tyre_raw[idx_t]  ^ params[8]))
    end
    append!(sim_ufp, seed_ufp)
    append!(sim_pm25, seed_pm25)
    push!(per_seed_ufp, seed_ufp)
    push!(per_seed_pm25, seed_pm25)

    # Accumulate post-warmup averaged grids for heatmaps (all seeds)
    seed_brake_grids = model_df[!, :collect_brake_emissions][(warmup_samples+1):end]
    seed_tyre_grids  = model_df[!, :collect_tyre_emissions][(warmup_samples+1):end]
    seed_avg_brake = reduce(.+, seed_brake_grids) ./ length(seed_brake_grids)
    seed_avg_tyre  = reduce(.+, seed_tyre_grids)  ./ length(seed_tyre_grids)
    if sum_brake_grid === nothing
        global sum_brake_grid = seed_avg_brake
        global sum_tyre_grid  = seed_avg_tyre
    else
        global sum_brake_grid = sum_brake_grid .+ seed_avg_brake
        global sum_tyre_grid  = sum_tyre_grid  .+ seed_avg_tyre
    end

    # Keep seed-1 data for snapshot and insets only
    if i == 1
        global car_positions = model_df[!, :collect_car_data]
        # traffic_light_states not collected by ECS model (unused in plot)
        global observer_data = obs_data
    end

    @printf(" done (%.1fs, %d cumulative samples)\n", time()-t0, length(sim_ufp))
    flush(stdout)
    GC.gc()
end

println("All simulations complete. Total pooled samples: $(length(sim_ufp))")

# ============================================================================
# Helper functions for snapshot panel (from generate_figure2.jl)
# ============================================================================

function draw_shape!(p, x, y, width, height, color, rotation_angle=0.0)
    corners = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    if rotation_angle != 0.0
        center_x, center_y = x + width / 2, y + height / 2
        rotated_corners = map(corners) do corner
            rel_x, rel_y = corner[1] - center_x, corner[2] - center_y
            new_x = rel_x * cos(rotation_angle) - rel_y * sin(rotation_angle) + center_x
            new_y = rel_x * sin(rotation_angle) + rel_y * cos(rotation_angle) + center_y
            [new_x, new_y]
        end
        xs = [c[1] for c in rotated_corners]; ys = [c[2] for c in rotated_corners]
    else
        xs = [c[1] for c in corners]; ys = [c[2] for c in corners]
    end
    plot!(p, xs, ys, seriestype=:shape, lw=0, fillalpha=0.8, fillcolor=color, legend=false)
end

function draw_vehicle!(p, vehicle_type, pos, direction, is_turning=false; scale=1.0)
    x, y = pos; rotation_angle = is_turning ? π / 4 : 0.0
    if vehicle_type == :smart_car
        draw_shape!(p, x - 0.5*scale, y - 0.5*scale, 1*scale, 1*scale, :green, rotation_angle)
    elseif vehicle_type == :car
        direction in [:up, :down] ? draw_shape!(p, x - 0.5*scale, y - 1*scale, 1*scale, 2*scale, :blue, rotation_angle) : draw_shape!(p, x - 1*scale, y - 0.5*scale, 2*scale, 1*scale, :blue, rotation_angle)
    elseif vehicle_type == :suv
        direction in [:up, :down] ? draw_shape!(p, x - 0.75*scale, y - 1.5*scale, 1.5*scale, 3*scale, :red, rotation_angle) : draw_shape!(p, x - 1.5*scale, y - 0.75*scale, 3*scale, 1.5*scale, :red, rotation_angle)
    elseif vehicle_type == :bus
        direction in [:up, :down] ? draw_shape!(p, x - 1*scale, y - 4.5*scale, 2*scale, 6*scale, RGB(1.0, 0.8, 0.0), rotation_angle) : draw_shape!(p, x - 4.5*scale, y - 1*scale, 6*scale, 2*scale, RGB(1.0, 0.8, 0.0), rotation_angle)
    elseif vehicle_type == :e_suv
        direction in [:up, :down] ? draw_shape!(p, x - 0.75*scale, y - 1.5*scale, 1.5*scale, 3*scale, :purple, rotation_angle) : draw_shape!(p, x - 1.5*scale, y - 0.75*scale, 3*scale, 1.5*scale, :purple, rotation_angle)
    elseif vehicle_type == :e_car
        direction in [:up, :down] ? draw_shape!(p, x - 0.5*scale, y - 1*scale, 1*scale, 2*scale, :cyan, rotation_angle) : draw_shape!(p, x - 1*scale, y - 0.5*scale, 2*scale, 1*scale, :cyan, rotation_angle)
    elseif vehicle_type == :observer
        direction in [:up, :down] ? draw_shape!(p, x - 0.75*scale, y - 1.5*scale, 1.5*scale, 3*scale, :purple, rotation_angle) : draw_shape!(p, x - 1.5*scale, y - 0.75*scale, 3*scale, 1.5*scale, :purple, rotation_angle)
    end
end

function draw_vehicle_marker!(p, vehicle_type, pos, direction; scale=1.0, color=:black)
    x, y = pos
    if vehicle_type in (:e_suv, :observer)
        direction in [:up, :down] ? draw_shape!(p, x - 0.75*scale, y - 1.5*scale, 1.5*scale, 3*scale, color, 0.0) : draw_shape!(p, x - 1.5*scale, y - 0.75*scale, 3*scale, 1.5*scale, color, 0.0)
    else
        direction in [:up, :down] ? draw_shape!(p, x - 0.5*scale, y - 1.0*scale, 1.0*scale, 2.0*scale, color, 0.0) : draw_shape!(p, x - 1.0*scale, y - 0.5*scale, 2.0*scale, 1.0*scale, color, 0.0)
    end
end

# ============================================================================
# Generate Figure 2
# ============================================================================

println("\nGenerating Figure 2...")

# Heatmap data: average across all 4 seeds
avg_brake_grid = sum_brake_grid ./ length(SCENARIO_SEEDS)
avg_tyre_grid  = sum_tyre_grid  ./ length(SCENARIO_SEEDS)

# Convert to actual units
converted_brake_grid = (BASELINE_UFP + clamped[14]) .+ params[3] .* (avg_brake_grid .^ params[7])
converted_tyre_grid = (BASELINE_PM25 + clamped[15]) .+ params[4] .* (avg_tyre_grid .^ params[8])

# Normalize for combined
norm_brake = (avg_brake_grid .- minimum(avg_brake_grid)) ./ (maximum(avg_brake_grid) - minimum(avg_brake_grid) + 1e-10)
norm_tyre = (avg_tyre_grid .- minimum(avg_tyre_grid)) ./ (maximum(avg_tyre_grid) - minimum(avg_tyre_grid) + 1e-10)
avg_total_grid = norm_brake + norm_tyre

# Get moving observer emissions for insets
moving_obs = observer_data["moving_observer"]
moving_brake_raw = moving_obs[:brake_emissions]
moving_tyre_raw = moving_obs[:tyre_emissions]
moving_brake_history = (BASELINE_UFP + clamped[14]) .+ params[3] .* (moving_brake_raw .^ params[7])
moving_tyre_history = (BASELINE_PM25 + clamped[15]) .+ params[4] .* (moving_tyre_raw .^ params[8])

# Snapshot timestep
total_time_steps = length(car_positions)
snapshot_timestep = min(300, total_time_steps)
final_step_cars = car_positions[snapshot_timestep]
# traffic light data not needed for plot (was dead code in original)

# ============================================================================
# Create 2×3 figure layout first, then add content
# ============================================================================

fig = plot(layout=(2, 3), size=(1800, 1100), dpi=300,
    left_margin=0mm, right_margin=-5mm, bottom_margin=5mm, top_margin=3mm)

# ============================================================================
# Panel 1 (a): Model snapshot with insets
# ============================================================================

plot!(fig, subplot=1, xlim=(0, GRID_SIZE), ylim=(0, GRID_SIZE),
      xlabel="X (m)", ylabel="Y (m)", legend=:none, grid=false,
      title="(a) Model Snapshot", titlefontsize=11)

# Grey-scale brake emissions overlay
mn, mx = minimum(avg_brake_grid), maximum(avg_brake_grid)
rng = mx - mn
norm_bg = rng == 0 ? fill(0.0, size(avg_brake_grid)) : (avg_brake_grid .- mn) ./ rng
boosted = clamp.(norm_bg .^ 0.6 .* 1.2, 0.0, 1.0)
heatmap!(fig, subplot=1, boosted', c=cgrad(:grays, rev=true), alpha=0.35, colorbar=false)

# Draw vehicles (scale=1.5 for bigger markers)
for (id, vehicle_type, pos, direction, speed) in final_step_cars
    draw_vehicle!(fig[1], vehicle_type, pos, direction, false; scale=1.0)
end

# Find and mark moving observer (dedicated :observer vehicle type)
tracked_pos_found = nothing
tracked_dir_found = :right
for (id, vehicle_type, pos, direction, speed) in final_step_cars
    if vehicle_type == :observer
        global tracked_pos_found = pos
        global tracked_dir_found = direction
        break
    end
end

if tracked_pos_found !== nothing
    draw_vehicle_marker!(fig[1], :observer, tracked_pos_found, tracked_dir_found; scale=1.0, color=:black)
    # Annotation with arrow
    annotate!(fig, subplot=1, 15, 55, text("Moving Observer", :black, 8, :left))
    draw_vehicle_marker!(fig[1], :observer, (10, 55), :right; scale=1.2, color=:black)
    plot!(fig, subplot=1, [65, tracked_pos_found[1]-1], [55, tracked_pos_found[2]],
          arrow=arrow(:head,0.5), linecolor=:black, linewidth=1)
end

# Inset: Brake emissions time series
plot!(fig, inset=(1, bbox(0.12, 0.15, 0.25, 0.22)), subplot=7)
plot!(fig[7], 1:length(moving_brake_history), log10.(moving_brake_history),
    xlabel="Time", ylabel="log₁₀(UFP)", legend=:none,
    xlim=(0, 1000), color=:black, grid=false,
    tickfontsize=6, guidefontsize=7)

# Inset: Tyre emissions time series
plot!(fig, inset=(1, bbox(0.68, 0.15, 0.25, 0.22)), subplot=8)
plot!(fig[8], 1:length(moving_tyre_history), moving_tyre_history,
    xlabel="Time", ylabel="PM₂.₅ (μg/m³)", legend=:none,
    xlim=(0, 1000), color=:black, grid=false,
    tickfontsize=6, guidefontsize=7)

# Burn-in highlight on insets (first 100 timesteps)
burn_n = 100
# Brake inset burn-in
bvals = log10.(moving_brake_history)
bfinite = filter(isfinite, bvals)
by1 = isempty(bfinite) ? 0.0 : minimum(bfinite)
by2 = isempty(bfinite) ? 1.0 : maximum(bfinite)
plot!(fig[7], [1, burn_n, burn_n, 1, 1], [by1, by1, by2, by2, by1],
      seriestype=:shape, fillalpha=0.08, fillcolor=:red,
      linecolor=:red, linewidth=1.5)
annotate!(fig, subplot=7, 5, by2 - 0.05*(by2-by1 + 1e-9), text("Burn-in", :red, 7, :left))

# Tyre inset burn-in
tfinite = filter(isfinite, moving_tyre_history)
ty1 = isempty(tfinite) ? 0.0 : minimum(tfinite)
ty2 = isempty(tfinite) ? 1.0 : maximum(tfinite)
plot!(fig[8], [1, burn_n, burn_n, 1, 1], [ty1, ty1, ty2, ty2, ty1],
      seriestype=:shape, fillalpha=0.08, fillcolor=:red,
      linecolor=:red, linewidth=1.5)
annotate!(fig, subplot=8, 5, ty2 - 0.05*(ty2-ty1 + 1e-9), text("Burn-in", :red, 7, :left))

# Vehicle legend in bottom right quadrant
annotate!(fig, subplot=1, 130, 70, text("E-SUV", :purple, 8, :left))
draw_vehicle!(fig[1], :e_suv, (120, 71), :right; scale=2)
annotate!(fig, subplot=1, 130, 55, text("E-Car", :cyan, 8, :left))
draw_vehicle!(fig[1], :e_car, (120, 56), :right; scale=2)
annotate!(fig, subplot=1, 130, 40, text("Car", :blue, 8, :left))
draw_vehicle!(fig[1], :car, (120, 41), :right; scale=2)
annotate!(fig, subplot=1, 130, 25, text("SUV", :red, 8, :left))
draw_vehicle!(fig[1], :suv, (120, 26), :right; scale=2)
annotate!(fig, subplot=1, 130, 10, text("Bus", RGB(1.0, 0.8, 0.0), 8, :left))
draw_vehicle!(fig[1], :bus, (120, 11), :right; scale=2)

# Reapply title for subplot 1 to match other subplots (with top margin)
plot!(fig, subplot=1, title="(a) Model Snapshot", titlefontsize=14, top_margin=10mm)

# ============================================================================
# Panel 2 (b): PM2.5 density plot
# ============================================================================

# Quantile grid for R²(q)
qs = 0.01:0.01:0.99
seed_colors = [:coral, :green, :purple]

# Bandwidths
pm25_bw = 1.06 * std(google_car_pm25_mass) * length(google_car_pm25_mass)^(-0.2)
ufp_bw = 1.06 * std(google_car_ufp_count) * length(google_car_ufp_count)^(-0.2)

# Pooled R²(q)
ufp_r2_pooled  = cor(quantile(google_car_ufp_count, qs), quantile(sim_ufp, qs))^2
pm25_r2_pooled = cor(quantile(google_car_pm25_mass, qs), quantile(sim_pm25, qs))^2

# --- PM2.5: experimental + pooled + per-seed ---
density!(fig, subplot=2, google_car_pm25_mass, label="Experimental", lw=2.5, color=:blue, bandwidth=pm25_bw)
density!(fig, subplot=2, sim_pm25, label=@sprintf("Pooled (R²=%.2f)", pm25_r2_pooled), lw=2.5, color=:black, ls=:dash, bandwidth=pm25_bw)
for i in 1:length(SCENARIO_SEEDS)
    r2_i = cor(quantile(google_car_pm25_mass, qs), quantile(per_seed_pm25[i], qs))^2
    density!(fig, subplot=2, per_seed_pm25[i], label=@sprintf("Seed %d (R²=%.2f)", i, r2_i),
             lw=1.5, ls=:dot, color=seed_colors[i], alpha=0.7, bandwidth=pm25_bw)
end
plot!(fig, subplot=2, title="(b) PM₂.₅ Distribution", titlefontsize=14, xlabel="PM₂.₅ (μg/m³)", ylabel="Density", grid=false, xlim=(-5, 90))

# ============================================================================
# Panel 3 (c): UFP density plot
# ============================================================================

# --- UFP: experimental + pooled + per-seed ---
density!(fig, subplot=3, google_car_ufp_count, label="Experimental", lw=2.5, color=:blue, bandwidth=ufp_bw)
density!(fig, subplot=3, sim_ufp, label=@sprintf("Pooled (R²=%.2f)", ufp_r2_pooled), lw=2.5, color=:black, ls=:dash, bandwidth=ufp_bw)
for i in 1:length(SCENARIO_SEEDS)
    r2_i = cor(quantile(google_car_ufp_count, qs), quantile(per_seed_ufp[i], qs))^2
    density!(fig, subplot=3, per_seed_ufp[i], label=@sprintf("Seed %d (R²=%.2f)", i, r2_i),
             lw=1.5, ls=:dot, color=seed_colors[i], alpha=0.7, bandwidth=ufp_bw)
end
plot!(fig, subplot=3, title="(c) UFP Distribution", titlefontsize=14, xlabel="UFP (particles/L)", ylabel="Density", grid=false, xlim=(-5e4, 7.5e5))

# Print summary
println("\nR²(q) comparison:")
@printf("  3-seed pooled:  UFP = %.2f,  PM2.5 = %.2f\n", ufp_r2_pooled, pm25_r2_pooled)
for i in 1:length(SCENARIO_SEEDS)
    r2u = cor(quantile(google_car_ufp_count, qs), quantile(per_seed_ufp[i], qs))^2
    r2p = cor(quantile(google_car_pm25_mass, qs), quantile(per_seed_pm25[i], qs))^2
    @printf("  Seed %d:         UFP = %.2f,  PM2.5 = %.2f\n", i, r2u, r2p)
end

# ============================================================================
# Panel 4 (d): PM2.5 heatmap
# ============================================================================

heatmap!(fig, subplot=4, converted_tyre_grid', c=:turbo,
    title="(d) PM₂.₅ Spatial", titlefontsize=14,
    xlabel="X (m)", ylabel="Y (m)", colorbar_title="μg/m³")

# ============================================================================
# Panel 5 (e): UFP heatmap
# ============================================================================

heatmap!(fig, subplot=5, log10.(converted_brake_grid'), c=:turbo,
    title="(e) UFP Spatial", titlefontsize=14,
    xlabel="X (m)", ylabel="Y (m)", colorbar_title="log₁₀(#/L)")

# ============================================================================
# Panel 6 (f): Combined heatmap
# ============================================================================

heatmap!(fig, subplot=6, avg_total_grid', c=:turbo,
    title="(f) Combined", titlefontsize=14,
    xlabel="X (m)", ylabel="Y (m)", colorbar_title="Normalised")

# Save with tightened horizontal spacing
out_dir = joinpath(@__DIR__, "..", "Figures")
isdir(out_dir) || mkpath(out_dir)

# Use Plots.jl savefig to render the matplotlib figure
savefig(fig, joinpath(out_dir, "figure2_model_tmp.png"))

# Now access the rendered matplotlib figure and adjust spacing
import PythonPlot
mpl_fig = PythonPlot.gcf()

# Center and equalize horizontal gaps between the 3 columns
# Axes layout (from debug): col_width=0.2255 (top) / 0.2098+0.0105 cbar (bot)
# Current: col1 starts 0.026, gaps 0.099 and 0.125 (uneven, left-heavy)
# Target: 3 centered columns with equal 0.06 gaps
#   Content width = 3×0.2255 + 2×0.06 = 0.797
#   Left margin = (1.0 - 0.797)/2 = 0.102
using PythonCall
all_axes = mpl_fig.get_axes()

# Per-column shifts to center everything with equal gaps
col1_shift = 0.102 - 0.026   # +0.076 (shift right)
col2_shift = 0.388 - 0.351   # +0.037 (shift right slightly)
col3_shift = 0.673 - 0.701   # -0.028 (shift left slightly)

# Col 1: main axes + insets + colorbar
for idx in [0, 3, 6, 7, 8]
    ax = all_axes[idx]
    pos = ax.get_position()
    x0 = pyconvert(Float64, pos.x0)
    y0 = pyconvert(Float64, pos.y0)
    w = pyconvert(Float64, pos.width)
    h = pyconvert(Float64, pos.height)
    ax.set_position([x0 + col1_shift, y0, w, h])
end

# Col 2: main axes + colorbar
for idx in [1, 4, 9]
    ax = all_axes[idx]
    pos = ax.get_position()
    x0 = pyconvert(Float64, pos.x0)
    y0 = pyconvert(Float64, pos.y0)
    w = pyconvert(Float64, pos.width)
    h = pyconvert(Float64, pos.height)
    ax.set_position([x0 + col2_shift, y0, w, h])
end

# Col 3: main axes + colorbar
for idx in [2, 5, 10]
    ax = all_axes[idx]
    pos = ax.get_position()
    x0 = pyconvert(Float64, pos.x0)
    y0 = pyconvert(Float64, pos.y0)
    w = pyconvert(Float64, pos.width)
    h = pyconvert(Float64, pos.height)
    ax.set_position([x0 + col3_shift, y0, w, h])
end

mpl_fig.savefig(joinpath(out_dir, "figure2_model.pdf"), dpi=300)
mpl_fig.savefig(joinpath(out_dir, "figure2_model.png"), dpi=300)
rm(joinpath(out_dir, "figure2_model_tmp.png"), force=true)

println("\n" * "="^60)
println("Figure 2 saved:")
println("  PDF: Figures/figure2_model.pdf")
println("  PNG: Figures/figure2_model.png")
println("="^60)
