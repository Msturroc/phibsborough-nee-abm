#!/usr/bin/env julia
"""
Figure 4: Speed x Variability Interaction -- 7x7 Annotated Heatmap (v2)

Uses quantile-ratio methodology: sort both baseline and scenario distributions,
divide element-wise, report (mean(ratio) - 1) × 100 as Δ%.
Threaded execution across grid cells (launch with julia -t 25).
"""

include("common_setup.jl")
using Plots
gr()

const FIGURES_DIR = joinpath(@__DIR__, "..", "Figures")
mkpath(FIGURES_DIR)

println("=" ^ 70)
println("FIGURE 4: SPEED x VARIABILITY 7x7 HEATMAP (v2 -- 4-seed)")
println("=" ^ 70)

# Grid: 7 speed levels x 7 variability levels (RECENTERED)
speeds_dms = [70, 80, 90, 105, 120, 130, 140]
vars_cms   = [10, 25, 37, 50, 75, 125, 200]  # recentered: baseline at index 4

speeds_ms  = speeds_dms ./ 10.0
vars_ms    = vars_cms ./ 100.0
speeds_kmh = round.(speeds_ms .* 3.6, digits=0)

speed_labels = [string(round(Int, s)) for s in speeds_kmh]
var_labels   = [string(round(v, digits=2)) for v in vars_ms]

n_speed = length(speeds_ms)
n_var   = length(vars_ms)

println("Grid: $(n_speed) speeds x $(n_var) variabilities = $(n_speed * n_var) cells")
println("Repeats per cell: $(N_REPEATS)")
println("Total simulations: $(n_speed * n_var * N_REPEATS)")

# Baseline cell indices
baseline_i = findfirst(==(105), speeds_dms)
baseline_j = findfirst(==(50), vars_cms)

# Run baseline cell first to get sorted reference vectors
println("Running baseline cell (speed=$(round(speeds_ms[baseline_i]*3.6, digits=0)) km/h, var=$(vars_ms[baseline_j]) m/s)...")
baseline_ufp_all  = Float64[]
baseline_pm25_all = Float64[]
for (rep, seed) in enumerate(SCENARIO_SEEDS)
    t0 = time()
    result = run_moving_observer(
        mean_speed=speeds_ms[baseline_i], speed_var=vars_ms[baseline_j], seed=seed
    )
    append!(baseline_ufp_all, result.ufp)
    append!(baseline_pm25_all, result.pm25)
    @printf("  Baseline rep %d/%d (seed=%d) done (%.1fs, %d samples)\n",
            rep, N_REPEATS, seed, time()-t0, length(result.ufp))
    flush(stdout)
    GC.gc()
end
baseline_ufp_sorted  = sort(baseline_ufp_all)
baseline_pm25_sorted = sort(baseline_pm25_all)
baseline_ufp  = mean(baseline_ufp_all)
baseline_pm25 = mean(baseline_pm25_all)
println("Baseline: UFP=$(round(Int, baseline_ufp)), PM2.5=$(round(baseline_pm25, digits=1))")
println("Baseline samples: $(length(baseline_ufp_sorted))")

# Run all cells in parallel using threads
ufp_means       = fill(NaN, n_speed, n_var)
pm25_means      = fill(NaN, n_speed, n_var)
ufp_ratio_means  = fill(NaN, n_speed, n_var)
pm25_ratio_means = fill(NaN, n_speed, n_var)

total_cells = n_speed * n_var
cells = [(i, j) for i in 1:n_speed for j in 1:n_var]

println("Using $(Threads.nthreads()) threads for $(total_cells) cells")

completed = Threads.Atomic{Int}(0)

Threads.@threads for idx in 1:length(cells)
    i, j = cells[idx]
    speed = speeds_ms[i]
    var = vars_ms[j]

    # Skip baseline cell — we already ran it
    if i == baseline_i && j == baseline_j
        ufp_means[i, j]       = baseline_ufp
        pm25_means[i, j]      = baseline_pm25
        ufp_ratio_means[i, j]  = 1.0
        pm25_ratio_means[i, j] = 1.0
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("[%d/%d] Speed=%.0f km/h, Var=%.2f m/s → BASELINE (reused)\n",
                c, total_cells, speed*3.6, var)
        flush(stdout)
        continue
    end

    ufp_all  = Float64[]
    pm25_all = Float64[]

    for (rep, seed) in enumerate(SCENARIO_SEEDS)
        t0 = time()
        result = run_moving_observer(
            mean_speed=speed, speed_var=var, seed=seed
        )
        append!(ufp_all, result.ufp)
        append!(pm25_all, result.pm25)
        GC.gc()
    end

    ufp_means[i, j]  = mean(ufp_all)
    pm25_means[i, j] = mean(pm25_all)

    # Quantile-ratio: sort scenario, divide element-wise by sorted baseline
    ufp_ratio_means[i, j]  = mean(sort(ufp_all) ./ baseline_ufp_sorted)
    pm25_ratio_means[i, j] = mean(sort(pm25_all) ./ baseline_pm25_sorted)

    c = Threads.atomic_add!(completed, 1) + 1
    @printf("[%d/%d] Speed=%.0f km/h, Var=%.2f m/s → UFP=%.0f, PM2.5=%.2f\n",
            c, total_cells, speed*3.6, var, ufp_means[i,j], pm25_means[i,j])
    flush(stdout)
end

# Compute % change from baseline using quantile-ratios
ufp_pct  = (ufp_ratio_means  .- 1.0) .* 100
pm25_pct = (pm25_ratio_means .- 1.0) .* 100

println("\nBaseline: UFP=$(round(Int, baseline_ufp)), PM2.5=$(round(baseline_pm25, digits=1))")

# ---- UFP Heatmap ----
println("\nCreating UFP heatmap...")

# Anchor the diverging colormap so that white = 0% exactly
ufp_cmin, ufp_cmax = -50, 75
ufp_zero_frac = (0 - ufp_cmin) / (ufp_cmax - ufp_cmin)
ufp_cmap = cgrad([:green, :white, :red], [0.0, ufp_zero_frac, 1.0])

p_ufp = heatmap(1:n_var, 1:n_speed, ufp_pct,
    c=ufp_cmap,
    clims=(ufp_cmin, ufp_cmax),
    xticks=(1:n_var, var_labels),
    yticks=(1:n_speed, speed_labels),
    xlabel="Speed Variability (m/s)",
    ylabel="Mean Speed (km/h)",
    title="(a) UFP % Change from Baseline",
    size=(700, 550), dpi=300,
    left_margin=10Plots.mm, bottom_margin=10Plots.mm,
    colorbar_title="% Change",
    aspect_ratio=:auto)

for i in 1:n_speed, j in 1:n_var
    val = ufp_pct[i, j]
    if !isnan(val)
        txt = @sprintf("%+.0f%%", val)
        col = abs(val) > 30 ? :white : :black
        fontsize = 8
        if speeds_dms[i] == 105 && vars_cms[j] == 50
            txt = "BASE"
            col = :black
            fontsize = 9
        end
        annotate!(j, i, text(txt, col, fontsize, :center))
    end
end

savefig(p_ufp, joinpath(FIGURES_DIR, "fig4_speed_var_ufp.pdf"))
savefig(p_ufp, joinpath(FIGURES_DIR, "fig4_speed_var_ufp.png"))
println("  Saved fig4_speed_var_ufp.pdf/png")

# ---- PM2.5 Heatmap ----
println("Creating PM2.5 heatmap...")

# Anchor the diverging colormap so that white = 0% exactly
pm_cmin, pm_cmax = -50, 110
pm_zero_frac = (0 - pm_cmin) / (pm_cmax - pm_cmin)
pm_cmap = cgrad([:green, :white, :red], [0.0, pm_zero_frac, 1.0])

p_pm25 = heatmap(1:n_var, 1:n_speed, pm25_pct,
    c=pm_cmap,
    clims=(pm_cmin, pm_cmax),
    xticks=(1:n_var, var_labels),
    yticks=(1:n_speed, speed_labels),
    xlabel="Speed Variability (m/s)",
    ylabel="Mean Speed (km/h)",
    title="(b) PM2.5 % Change from Baseline",
    size=(700, 550), dpi=300,
    left_margin=10Plots.mm, bottom_margin=10Plots.mm,
    colorbar_title="% Change",
    aspect_ratio=:auto)

for i in 1:n_speed, j in 1:n_var
    val = pm25_pct[i, j]
    if !isnan(val)
        txt = @sprintf("%+.0f%%", val)
        col = abs(val) > 40 ? :white : :black
        fontsize = 8
        if speeds_dms[i] == 105 && vars_cms[j] == 50
            txt = "BASE"
            col = :black
            fontsize = 9
        end
        annotate!(j, i, text(txt, col, fontsize, :center))
    end
end

savefig(p_pm25, joinpath(FIGURES_DIR, "fig4_speed_var_pm25.pdf"))
savefig(p_pm25, joinpath(FIGURES_DIR, "fig4_speed_var_pm25.png"))
println("  Saved fig4_speed_var_pm25.pdf/png")

# Print summary grid
println("\n" * "=" ^ 70)
println("UFP % CHANGE GRID")
println("=" ^ 70)
print(rpad("Speed\\Var", 10))
for v in vars_ms; print(rpad(string(v), 8)); end
println()
for (i, s) in enumerate(speeds_kmh)
    print(rpad("$(round(Int, s)) km/h", 10))
    for j in 1:n_var
        print(rpad(@sprintf("%+.0f%%", ufp_pct[i,j]), 8))
    end
    println()
end

println("\n" * "=" ^ 70)
println("PM2.5 % CHANGE GRID")
println("=" ^ 70)
print(rpad("Speed\\Var", 10))
for v in vars_ms; print(rpad(string(v), 8)); end
println()
for (i, s) in enumerate(speeds_kmh)
    print(rpad("$(round(Int, s)) km/h", 10))
    for j in 1:n_var
        print(rpad(@sprintf("%+.0f%%", pm25_pct[i,j]), 8))
    end
    println()
end

# Print absolute values for paper text
println("\n" * "=" ^ 70)
println("ABSOLUTE VALUES")
println("=" ^ 70)
println("Slow+smooth corner (25 km/h, 0.10 m/s): UFP=$(round(Int, ufp_means[1,1])), PM2.5=$(round(pm25_means[1,1], digits=1))")
println("Fast+erratic corner (50 km/h, 2.00 m/s): UFP=$(round(Int, ufp_means[end,end])), PM2.5=$(round(pm25_means[end,end], digits=1))")
println("Baseline (38 km/h, 0.50 m/s): UFP=$(round(Int, baseline_ufp)), PM2.5=$(round(baseline_pm25, digits=1))")

# Super-additivity check
speed_only = ufp_pct[1, baseline_j]
var_only   = ufp_pct[baseline_i, 1]
combined   = ufp_pct[1, 1]
@printf("\nSuper-additivity check (UFP):\n")
@printf("  Speed only (25 km/h, baseline var): %+.1f%%\n", speed_only)
@printf("  Var only (baseline speed, 0.10 m/s): %+.1f%%\n", var_only)
@printf("  Sum of individual: %+.1f%%\n", speed_only + var_only)
@printf("  Combined (25 km/h, 0.10 m/s): %+.1f%%\n", combined)
@printf("  Super-additive bonus: %.1f pp\n", combined - (speed_only + var_only))

speed_only_pm = pm25_pct[1, baseline_j]
var_only_pm   = pm25_pct[baseline_i, 1]
combined_pm   = pm25_pct[1, 1]
@printf("\nSuper-additivity check (PM2.5):\n")
@printf("  Speed only (25 km/h, baseline var): %+.1f%%\n", speed_only_pm)
@printf("  Var only (baseline speed, 0.10 m/s): %+.1f%%\n", var_only_pm)
@printf("  Sum of individual: %+.1f%%\n", speed_only_pm + var_only_pm)
@printf("  Combined (25 km/h, 0.10 m/s): %+.1f%%\n", combined_pm)
@printf("  Super-additive bonus: %.1f pp\n", combined_pm - (speed_only_pm + var_only_pm))

println("\nFigure 4 complete!")
