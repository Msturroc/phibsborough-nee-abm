#!/usr/bin/env julia
"""
make_readme_animation.jl

Generate an animated GIF of the Phibsborough traffic simulation for the
repository README. Matches Figure 2 panel (a) exactly: PythonPlot backend,
white background, greyscale brake emissions overlay with alpha=0.35,
coloured vehicles, vehicle legend in bottom-right with drawn icons,
observer icon + label, and two observer time-series INSETS overlaid at the
same bbox positions as Figure 2 with burn-in highlight.

Uses phibsborough_ecs_v2.jl with v7 calibrated parameters.

Usage:  julia make_readme_animation.jl
"""

using StatsPlots
pythonplot()
using Colors
using Printf
using Statistics
using DelimitedFiles

const GRID_SIZE = 200
ENV["GRID_SIZE"] = string(GRID_SIZE)

println("Loading ECS model v2...")
include(joinpath(@__DIR__, "..", "src", "phibsborough_ecs_v2.jl"))

# ── Load calibrated parameters ──────────────────────────────────────────
param_file = joinpath(@__DIR__, "..", "data", "best_parameters_cmaes_v7_excellent_contender.txt")
println("Loading parameters from: $param_file")
log10_params = vec(readdlm(param_file))

const LOWER = Float64[log10(8.0), log10(0.5), log10(0.001), log10(0.01), log10(0.3),
                       log10(0.3), log10(0.5), log10(0.3), log10(10.0), log10(1.0),
                       log10(15.0), 1.0, 1.0, -2000.0, -0.5]
const UPPER = Float64[log10(17.5), log10(5.0), log10(1e7), log10(500.0), log10(5.0),
                       log10(200.0), log10(5.0), log10(6.0), log10(20.0), log10(10.0),
                       log10(90.0), 6.0, 6.0, 2000.0, 2.0]

clamped = [clamp(log10_params[i], LOWER[i], UPPER[i]) for i in 1:15]
params = 10 .^ clamped[1:11]

# ── Simulation setup (matching Figure 2 exactly: 1100s total) ───────────
dt = calculate_adaptive_dt(params[1], speed_variability=params[2])
const TOTAL_TIME  = 1100.0
const WARMUP_TIME = 100.0
n_steps     = round(Int, TOTAL_TIME / dt)
dsi         = max(1, round(Int, 1.0 / dt))
warmup_samp = round(Int, (WARMUP_TIME / dt) / dsi)

@printf("dt=%.4fs  steps=%d  sample_interval=%d  warmup_samples=%d\n",
        dt, n_steps, dsi, warmup_samp)

# ── Run simulation ──────────────────────────────────────────────────────
println("Running simulation (seed 92345)...")
t0 = time()
model_df, obs_data = run_simulation(;
    dims = (GRID_SIZE, GRID_SIZE),
    mean_speed = params[1], speed_variability = params[2],
    brake_decay_rate = params[5], tyre_decay_rate = params[6],
    max_sight_distance = round(Int, params[9]),
    planning_distance = params[10],
    random_seed = 92345,
    green_duration_s = params[11], amber_duration_s = 3.0,
    red_duration_s = params[11] + 3.0,
    brake_dispersion_radius = clamp(round(Int, clamped[12]), 1, 6),
    tyre_dispersion_radius = clamp(round(Int, clamped[13]), 1, 6),
    tracked_vehicle_type = :observer,
    steps = n_steps, data_sample_interval = dsi,
)
@printf("Simulation done in %.1fs\n", time() - t0)

# ── Extract post-warmup data ────────────────────────────────────────────
cars   = model_df[!, :collect_car_data][(warmup_samp+1):end]
brakes = model_df[!, :collect_brake_emissions][(warmup_samp+1):end]
n_total = length(brakes)
println("Post-warmup frames: $n_total")

# Observer time series (moving observer — same as Figure 2)
moving_obs = obs_data["moving_observer"]
obs_brake_raw = moving_obs[:brake_emissions][(warmup_samp+1):end]
obs_tyre_raw  = moving_obs[:tyre_emissions][(warmup_samp+1):end]

# Convert to physical units (matching Figure 2)
obs_brake_phys = (clamped[14]) .+ params[3] .* (obs_brake_raw .^ params[7])
obs_tyre_phys  = (clamped[15]) .+ params[4] .* (obs_tyre_raw  .^ params[8])

# ── Cumulative average brake grid for background ────────────────────────
# (Figure 2 uses time-averaged brake grid; here we build it up over time)
cum_brake = zeros(GRID_SIZE, GRID_SIZE)
avg_brake_frames = Vector{Matrix{Float64}}(undef, n_total)
for i in 1:n_total
    cum_brake .+= brakes[i]
    avg_brake_frames[i] = cum_brake ./ i
end

# ── Pre-compute inset ylims (from FULL time series, matching Figure 2) ──
brake_vals = log10.(obs_brake_phys)
brake_finite = filter(isfinite, brake_vals)
by1 = isempty(brake_finite) ? 0.0 : minimum(brake_finite)
by2 = isempty(brake_finite) ? 1.0 : maximum(brake_finite)

tyre_finite = filter(isfinite, obs_tyre_phys)
ty1 = isempty(tyre_finite) ? 0.0 : minimum(tyre_finite)
ty2 = isempty(tyre_finite) ? 1.0 : maximum(tyre_finite)

# Burn-in length (matching Figure 2: first 100 post-warmup timesteps)
const BURN_N = 100

# ── Vehicle drawing (matching Figure 2 exactly) ─────────────────────────
function draw_shape_anim!(p, x, y, width, height, color)
    xs = [x, x + width, x + width, x, x]
    ys = [y, y, y + height, y + height, y]
    plot!(p, xs, ys, seriestype=:shape, lw=0, fillalpha=0.8, fillcolor=color, label=false)
end

function draw_vehicle_anim!(p, vehicle_type, pos, direction; scale=1.0)
    x, y = pos
    if vehicle_type == :smart_car || vehicle_type == :microcar
        draw_shape_anim!(p, x - 0.5*scale, y - 0.5*scale, 1*scale, 1*scale, :green)
    elseif vehicle_type == :car
        direction in [:up, :down] ?
            draw_shape_anim!(p, x - 0.5*scale, y - 1*scale, 1*scale, 2*scale, :blue) :
            draw_shape_anim!(p, x - 1*scale, y - 0.5*scale, 2*scale, 1*scale, :blue)
    elseif vehicle_type == :e_car
        direction in [:up, :down] ?
            draw_shape_anim!(p, x - 0.5*scale, y - 1*scale, 1*scale, 2*scale, :cyan) :
            draw_shape_anim!(p, x - 1*scale, y - 0.5*scale, 2*scale, 1*scale, :cyan)
    elseif vehicle_type == :suv
        direction in [:up, :down] ?
            draw_shape_anim!(p, x - 0.75*scale, y - 1.5*scale, 1.5*scale, 3*scale, :red) :
            draw_shape_anim!(p, x - 1.5*scale, y - 0.75*scale, 3*scale, 1.5*scale, :red)
    elseif vehicle_type == :e_suv
        direction in [:up, :down] ?
            draw_shape_anim!(p, x - 0.75*scale, y - 1.5*scale, 1.5*scale, 3*scale, :purple) :
            draw_shape_anim!(p, x - 1.5*scale, y - 0.75*scale, 3*scale, 1.5*scale, :purple)
    elseif vehicle_type == :observer
        direction in [:up, :down] ?
            draw_shape_anim!(p, x - 0.75*scale, y - 1.5*scale, 1.5*scale, 3*scale, :black) :
            draw_shape_anim!(p, x - 1.5*scale, y - 0.75*scale, 3*scale, 1.5*scale, :black)
    elseif vehicle_type == :bus
        direction in [:up, :down] ?
            draw_shape_anim!(p, x - 1*scale, y - 4.5*scale, 2*scale, 6*scale, RGB(1.0, 0.8, 0.0)) :
            draw_shape_anim!(p, x - 4.5*scale, y - 1*scale, 6*scale, 2*scale, RGB(1.0, 0.8, 0.0))
    end
end

# ── Build animation ─────────────────────────────────────────────────────
# ~1000 post-warmup samples; subsample to keep GIF manageable
frame_step = 10   # every 10s → ~100 frames
idxs = 1:frame_step:n_total
nf = length(idxs)
println("Animating $nf frames @ 10 fps → $(round(nf/10, digits=1))s GIF...")

anim = @animate for (fi, i) in enumerate(idxs)
    # ── Greyscale brake emission background (Figure 2 style) ──
    avg_grid = avg_brake_frames[i]
    mn, mx = minimum(avg_grid), maximum(avg_grid)
    rng = mx - mn
    norm_bg = rng == 0 ? fill(0.0, size(avg_grid)) : (avg_grid .- mn) ./ rng
    boosted = clamp.(norm_bg .^ 0.6 .* 1.2, 0.0, 1.0)

    # Base plot (white background, axes)
    p = plot(;
        xlim=(0, GRID_SIZE), ylim=(0, GRID_SIZE),
        aspect_ratio=:equal, grid=false,
        xlabel="X (m)", ylabel="Y (m)",
        legend=:none, framestyle=:box,
        background_color=:white,
        title=@sprintf("Phibsborough Intersection  —  t = %ds", i-1),
        titlefontsize=11, tickfontsize=7, guidefontsize=8,
        size=(700, 700),
    )

    # Greyscale heatmap overlay with alpha=0.35 (matching Figure 2 exactly)
    heatmap!(p, boosted'; c=cgrad(:grays, rev=true), alpha=0.35, colorbar=false)

    # ── Draw all vehicles ──
    for (_, vt, pos, dir, _) in cars[i]
        draw_vehicle_anim!(p, vt, pos, dir; scale=1.0)
    end

    # ── Vehicle legend — bottom-right quadrant (Figure 2 positions) ──
    annotate!(p, 130, 70, text("E-SUV", :purple, 8, :left))
    draw_vehicle_anim!(p, :e_suv, (120, 71), :right; scale=2)
    annotate!(p, 130, 55, text("E-Car", :cyan, 8, :left))
    draw_vehicle_anim!(p, :e_car, (120, 56), :right; scale=2)
    annotate!(p, 130, 40, text("Car", :blue, 8, :left))
    draw_vehicle_anim!(p, :car, (120, 41), :right; scale=2)
    annotate!(p, 130, 25, text("SUV", :red, 8, :left))
    draw_vehicle_anim!(p, :suv, (120, 26), :right; scale=2)
    annotate!(p, 130, 10, text("Bus", RGB(1.0, 0.8, 0.0), 8, :left))
    draw_vehicle_anim!(p, :bus, (120, 11), :right; scale=2)

    # ── Observer label with icon (Figure 2 style) ──
    annotate!(p, 15, 55, text("Moving Observer", :black, 8, :left))
    draw_vehicle_anim!(p, :observer, (10, 55), :right; scale=1.2)

    # ── Inset: Brake (UFP) time series — Figure 2 bbox exactly ──
    plot!(p, inset=(1, bbox(0.12, 0.15, 0.25, 0.22)), subplot=2)
    plot!(p[2], 1:i, brake_vals[1:i];
        xlabel="Time", ylabel="log₁₀(UFP)",
        legend=:none, color=:black, grid=false,
        xlim=(0, 1000), ylim=(by1, by2),
        tickfontsize=6, guidefontsize=7, linewidth=0.8)
    # Burn-in highlight (Figure 2: red box over first 100 timesteps)
    plot!(p[2], [1, BURN_N, BURN_N, 1, 1], [by1, by1, by2, by2, by1],
          seriestype=:shape, fillalpha=0.08, fillcolor=:red,
          linecolor=:red, linewidth=1.5)
    annotate!(p, subplot=2, 5, by2 - 0.05*(by2-by1 + 1e-9), text("Burn-in", :red, 7, :left))

    # ── Inset: Tyre (PM2.5) time series — Figure 2 bbox exactly ──
    plot!(p, inset=(1, bbox(0.68, 0.15, 0.25, 0.22)), subplot=3)
    plot!(p[3], 1:i, obs_tyre_phys[1:i];
        xlabel="Time", ylabel="PM₂.₅ (μg/m³)",
        legend=:none, color=:black, grid=false,
        xlim=(0, 1000), ylim=(ty1, ty2),
        tickfontsize=6, guidefontsize=7, linewidth=0.8)
    # Burn-in highlight
    plot!(p[3], [1, BURN_N, BURN_N, 1, 1], [ty1, ty1, ty2, ty2, ty1],
          seriestype=:shape, fillalpha=0.08, fillcolor=:red,
          linecolor=:red, linewidth=1.5)
    annotate!(p, subplot=3, 5, ty2 - 0.05*(ty2-ty1 + 1e-9), text("Burn-in", :red, 7, :left))

    fi % 10 == 0 && @printf("  frame %d/%d\n", fi, nf)
end

# ── Save ────────────────────────────────────────────────────────────────
outpath = joinpath(@__DIR__, "..", "Figures", "readme_animation.gif")
gif(anim, outpath, fps=10)
println("\n✓ Saved: $outpath")
sz = filesize(outpath) / 1024^2
@printf("  Size: %.1f MB\n", sz)
if sz > 10
    println("  Tip: compress with gifsicle:")
    opt_path = joinpath(@__DIR__, "..", "Figures", "readme_animation_opt.gif")
    println("    gifsicle -O3 --colors 128 --lossy=80 $outpath -o $opt_path")
end
