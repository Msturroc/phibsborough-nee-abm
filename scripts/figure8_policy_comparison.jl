#!/usr/bin/env julia
"""
Figure 8: Policy Comparison — Bar chart comparing % change in moving-observer
NEE exposure from baseline for all interventions.

Per-seed quantile-ratio methodology: for each seed, sort baseline and scenario
distributions, divide element-wise, take mean. Then average ratios across seeds
so each seed has equal weight (prevents outlier tails from dominating).

Uses 8 seeds for inline scenarios (speed limit, smooth traffic) to ensure
robust estimates for these small-effect interventions. Modal shift and
electrification data loaded from pre-computed summaries (figures 5-7).
"""

include("common_setup.jl")
using Plots
gr()

const FIGURES_DIR = joinpath(@__DIR__, "..", "Figures")
const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(FIGURES_DIR)

println("=" ^ 70)
println("FIGURE 8: POLICY COMPARISON (v2 — 8-seed inline, per-seed ratios)")
println("=" ^ 70)

# ============================================================================
# Speed limit and smooth traffic scenarios (run inline, 8-seed per-seed ratios)
# ============================================================================

println("\n--- Running speed/variability scenarios (8-seed per-seed quantile-ratio) ---")

speed_30 = 30.0 / 3.6
smooth_var = SPEED_VAR * 0.50

# 8 seeds for inline scenarios: the standard 4 plus 4 additional for robustness.
# Small-effect scenarios (smooth traffic) are noisy with 4 seeds due to
# chaotic divergence in traffic dynamics; 8 seeds provides a clearer signal.
const INLINE_SEEDS = [12345, 22345, 32345, 42345, 52345, 62345, 72345, 82345]

@printf("  Baseline: %.0f km/h, var=%.2f m/s\n", MEAN_SPEED*3.6, SPEED_VAR)
@printf("  30 km/h: %.1f m/s\n", speed_30)
@printf("  Smooth: var=%.2f m/s (50%% of calibrated %.2f m/s)\n", smooth_var, SPEED_VAR)
@printf("  Seeds: %d (%s)\n", length(INLINE_SEEDS), join(INLINE_SEEDS, ", "))

seed_ufp_30 = Float64[]
seed_pm25_30 = Float64[]
seed_ufp_smooth = Float64[]
seed_pm25_smooth = Float64[]

for (rep, seed) in enumerate(INLINE_SEEDS)
    @printf("  Seed %d (%d/%d)...\n", seed, rep, length(INLINE_SEEDS))
    flush(stdout)

    # Baseline for this seed
    t0 = time()
    base = run_moving_observer(seed=seed)
    base_ufp_sorted  = sort(base.ufp)
    base_pm25_sorted = sort(base.pm25)
    @printf("    Baseline: UFP=%.0f, PM2.5=%.2f (%.1fs)\n",
            mean(base.ufp), mean(base.pm25), time()-t0)

    # 30 km/h for this seed
    t0 = time()
    r30 = run_moving_observer(seed=seed, mean_speed=speed_30)
    push!(seed_ufp_30,  mean(sort(r30.ufp) ./ base_ufp_sorted))
    push!(seed_pm25_30, mean(sort(r30.pm25) ./ base_pm25_sorted))
    @printf("    30 km/h:  UFP ratio=%.4f, PM2.5 ratio=%.4f (%.1fs)\n",
            seed_ufp_30[end], seed_pm25_30[end], time()-t0)

    # Smooth traffic for this seed
    t0 = time()
    rsmooth = run_moving_observer(seed=seed, speed_var=smooth_var)
    push!(seed_ufp_smooth,  mean(sort(rsmooth.ufp) ./ base_ufp_sorted))
    push!(seed_pm25_smooth, mean(sort(rsmooth.pm25) ./ base_pm25_sorted))
    @printf("    Smooth:   UFP ratio=%.4f, PM2.5 ratio=%.4f (%.1fs)\n",
            seed_ufp_smooth[end], seed_pm25_smooth[end], time()-t0)

    flush(stdout)
    GC.gc()
end

# Average across seeds (each seed has equal weight)
ufp_30_ratio    = mean(seed_ufp_30)
pm25_30_ratio   = mean(seed_pm25_30)
ufp_smooth_ratio  = mean(seed_ufp_smooth)
pm25_smooth_ratio = mean(seed_pm25_smooth)

println("\n  Per-seed ratios ($(length(INLINE_SEEDS)) seeds):")
@printf("    30 km/h  UFP:  %s → mean=%.4f\n",
        join([@sprintf("%.3f", r) for r in seed_ufp_30], ", "), ufp_30_ratio)
@printf("    30 km/h  PM25: %s → mean=%.4f\n",
        join([@sprintf("%.3f", r) for r in seed_pm25_30], ", "), pm25_30_ratio)
@printf("    Smooth   UFP:  %s → mean=%.4f\n",
        join([@sprintf("%.3f", r) for r in seed_ufp_smooth], ", "), ufp_smooth_ratio)
@printf("    Smooth   PM25: %s → mean=%.4f\n",
        join([@sprintf("%.3f", r) for r in seed_pm25_smooth], ", "), pm25_smooth_ratio)

# ============================================================================
# Load modal shift summaries (from figures 6 and 7)
# ============================================================================

println("\n--- Loading modal shift data ---")

bus_summary = CSV.read(joinpath(DATA_DIR, "bus_modal_summary.csv"), DataFrame)
micro_summary = CSV.read(joinpath(DATA_DIR, "micro_modal_summary.csv"), DataFrame)

# Use mean quantile-ratios for consistency with ratio-based tables and figures
bus_25_ufp_ratio  = bus_summary[bus_summary.scenario .== "25% to Bus", :ufp_ratio_mean][1]
bus_25_pm25_ratio = bus_summary[bus_summary.scenario .== "25% to Bus", :pm25_ratio_mean][1]

micro_25_ufp_ratio  = micro_summary[micro_summary.scenario .== "25% to Microcar", :ufp_ratio_mean][1]
micro_25_pm25_ratio = micro_summary[micro_summary.scenario .== "25% to Microcar", :pm25_ratio_mean][1]

bus_25_ufp_pct  = (bus_25_ufp_ratio - 1.0) * 100
bus_25_pm25_pct = (bus_25_pm25_ratio - 1.0) * 100
micro_25_ufp_pct  = (micro_25_ufp_ratio - 1.0) * 100
micro_25_pm25_pct = (micro_25_pm25_ratio - 1.0) * 100

# ============================================================================
# Load electrification summary (from figure 5)
# ============================================================================

println("\n--- Loading electrification data ---")

elec_summary = CSV.read(joinpath(DATA_DIR, "electrification_summary.csv"), DataFrame)

elec_50_regen_ufp_ratio  = elec_summary[elec_summary.scenario .== "50% Electric (with regen)", :ufp_ratio_mean][1]
elec_50_regen_pm25_ratio = elec_summary[elec_summary.scenario .== "50% Electric (with regen)", :pm25_ratio_mean][1]

elec_50_ufp_pct  = (elec_50_regen_ufp_ratio - 1.0) * 100
elec_50_pm25_pct = (elec_50_regen_pm25_ratio - 1.0) * 100

@printf("    50%% Electrified (with regen): UFP=%+.1f%%, PM2.5=%+.1f%%\n", elec_50_ufp_pct, elec_50_pm25_pct)

# ============================================================================
# Combined policy scenario (run inline, 8-seed per-seed ratios)
# ============================================================================

println("\n--- Running combined policy scenario (8-seed per-seed quantile-ratio) ---")
println("  Combo fleet (per direction): cars=3, ecars=4, suvs=5, esuvs=5, buses=2, micros=2")
println("  Speed: 30 km/h, smooth traffic (var=50%), regen=0.8")
println("  (= 25% bus shift + 25% cars→microcars + 50% electrification + 30 km/h + smooth)")

seed_ufp_combo = Float64[]
seed_pm25_combo = Float64[]

for (rep, seed) in enumerate(INLINE_SEEDS)
    @printf("  Seed %d (%d/%d)...\n", seed, rep, length(INLINE_SEEDS))
    flush(stdout)

    # Baseline for this seed
    t0 = time()
    base = run_moving_observer(seed=seed)
    base_ufp_sorted  = sort(base.ufp)
    base_pm25_sorted = sort(base.pm25)

    # Combined policy
    t0 = time()
    rcombo = run_moving_observer(seed=seed,
        mean_speed=speed_30, speed_var=smooth_var,
        n_cars=3, n_ecars=4, n_suvs=5, n_esuvs=5,
        n_buses=2, n_micros=2,
        bus_occ=15.1)
    push!(seed_ufp_combo,  mean(sort(rcombo.ufp) ./ base_ufp_sorted))
    push!(seed_pm25_combo, mean(sort(rcombo.pm25) ./ base_pm25_sorted))
    @printf("    Combo:    UFP ratio=%.4f, PM2.5 ratio=%.4f (%.1fs)\n",
            seed_ufp_combo[end], seed_pm25_combo[end], time()-t0)

    flush(stdout)
    GC.gc()
end

ufp_combo_ratio  = mean(seed_ufp_combo)
pm25_combo_ratio = mean(seed_pm25_combo)

println("\n  Per-seed ratios ($(length(INLINE_SEEDS)) seeds):")
@printf("    Combo    UFP:  %s → mean=%.4f\n",
        join([@sprintf("%.3f", r) for r in seed_ufp_combo], ", "), ufp_combo_ratio)
@printf("    Combo    PM25: %s → mean=%.4f\n",
        join([@sprintf("%.3f", r) for r in seed_pm25_combo], ", "), pm25_combo_ratio)

combo_ufp_pct  = (ufp_combo_ratio - 1.0) * 100
combo_pm25_pct = (pm25_combo_ratio - 1.0) * 100

# ============================================================================
# Build policy comparison data
# ============================================================================

policies = [
    "30 km/h\nspeed limit",
    "Smooth traffic\n(low variability)",
    "25% Bus\nmodal shift",
    "25% Microcar\nmodal shift",
    "50% Electrified\n(with regen)",
    "Combined\npolicy",
]

ufp_pct = [
    (ufp_30_ratio - 1.0) * 100,
    (ufp_smooth_ratio - 1.0) * 100,
    bus_25_ufp_pct,
    micro_25_ufp_pct,
    elec_50_ufp_pct,
    combo_ufp_pct,
]

pm25_pct = [
    (pm25_30_ratio - 1.0) * 100,
    (pm25_smooth_ratio - 1.0) * 100,
    bus_25_pm25_pct,
    micro_25_pm25_pct,
    elec_50_pm25_pct,
    combo_pm25_pct,
]

# ============================================================================
# Create grouped bar chart
# ============================================================================

println("\nCreating policy comparison chart...")

n = length(policies)
x = 1:n
bar_width = 0.35

p = plot(size=(900, 500), dpi=300,
    left_margin=10Plots.mm, bottom_margin=15Plots.mm,
    top_margin=5Plots.mm, right_margin=5Plots.mm)

bar!(x .- bar_width/2, ufp_pct, bar_width=bar_width,
     label="UFP (brake wear)", color=:steelblue)
bar!(x .+ bar_width/2, pm25_pct, bar_width=bar_width,
     label="PM₂.₅ (tyre wear)", color=:forestgreen)

hline!([0], color=:grey, linewidth=1.5, label="", alpha=0.7)

plot!(xticks=(x, policies), xrotation=0,
      ylabel="Δ% Mean Moving-Observer Exposure",
      title="Policy Comparison: Change in NEE Exposure",
      legend=:topleft, grid=false,
      ylim=(minimum(vcat(ufp_pct, pm25_pct)) - 10, maximum(vcat(ufp_pct, pm25_pct)) + 10))

savefig(p, joinpath(FIGURES_DIR, "fig7_policy_comparison.pdf"))
savefig(p, joinpath(FIGURES_DIR, "fig7_policy_comparison.png"))
println("  Saved fig7_policy_comparison.pdf/png")

# Print summary
println("\n" * "=" ^ 70)
println("POLICY COMPARISON SUMMARY")
println("=" ^ 70)
println(rpad("Policy", 30), rpad("UFP Δ%", 12), "PM2.5 Δ%")
println("-" ^ 55)
for i in 1:n
    println(rpad(replace(policies[i], "\n" => " "), 30),
            rpad(@sprintf("%+.1f%%", ufp_pct[i]), 12),
            @sprintf("%+.1f%%", pm25_pct[i]))
end

println("\nFigure 8 complete!")
