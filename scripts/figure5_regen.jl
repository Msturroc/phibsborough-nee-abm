#!/usr/bin/env julia
"""
Figure 5: Electrification × Regenerative Braking Uncertainty (v2)

Quantile-ratio approach: sort baseline and scenario distributions, divide
element-wise. Deviations from 1.0 show the distributional policy effect.

Sweeps 25%, 50%, 75% electrification, each with and without regen.
"""

include("common_setup.jl")
using TidierPlots
using CategoricalArrays
using StatsBase

const FIGURES_DIR = joinpath(@__DIR__, "..", "Figures")
const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(FIGURES_DIR)
mkpath(DATA_DIR)

println("=" ^ 70)
println("FIGURE 5: REGEN × ELECTRIFICATION (quantile-ratio, 4-seed)")
println("=" ^ 70)

# ============================================================================
# Fleet calculator
# ============================================================================

function calculate_electrified_fleet(elec_pct::Float64)
    # Work in per-direction terms to avoid div() truncation
    cars_to_shift = round(Int, N_CARS_PER_DIR * elec_pct)
    suvs_to_shift = round(Int, N_SUVS_PER_DIR * elec_pct)
    n_ecars = N_ECARS_PER_DIR + cars_to_shift
    n_esuvs = N_ESUVS_PER_DIR + suvs_to_shift
    n_cars  = N_CARS_PER_DIR - cars_to_shift
    n_suvs  = N_SUVS_PER_DIR - suvs_to_shift
    total_per_dir = n_cars + n_ecars + n_suvs + n_esuvs + N_BUSES_PER_DIR
    @assert total_per_dir == 25 "Fleet calculator error: expected 25 vehicles/dir, got $total_per_dir"
    return (
        n_cars  = n_cars,
        n_ecars = n_ecars,
        n_suvs  = n_suvs,
        n_esuvs = n_esuvs,
    )
end

# ============================================================================
# Scenario definitions
# ============================================================================

elec_levels = [
    ("25% Electric", 0.25),
    ("50% Electric", 0.50),
    ("75% Electric", 0.75),
]

# Print fleet configurations with mass accounting
println("\nFLEET CONFIGURATIONS (per direction)")
println("=" ^ 70)

baseline_vehicles = N_CARS_PER_DIR + N_ECARS_PER_DIR + N_SUVS_PER_DIR + N_ESUVS_PER_DIR + N_BUSES_PER_DIR
baseline_mass = N_CARS_PER_DIR * 1500.0 + N_ECARS_PER_DIR * 1700.0 +
                N_SUVS_PER_DIR * 2200.0 + N_ESUVS_PER_DIR * 2600.0 +
                N_BUSES_PER_DIR * (12000.0 + DEFAULT_BUS_OCC * 75.0)
baseline_pax = (N_CARS_PER_DIR + N_ECARS_PER_DIR + N_SUVS_PER_DIR + N_ESUVS_PER_DIR) * PASSENGERS_PER_CAR +
               N_BUSES_PER_DIR * DEFAULT_BUS_OCC
@printf("  Baseline             → cars=%d, ecars=%d, suvs=%d, esuvs=%d, buses=%d | vehicles=%d, mass=%.1f t, kg/pax=%.0f\n",
        N_CARS_PER_DIR, N_ECARS_PER_DIR, N_SUVS_PER_DIR, N_ESUVS_PER_DIR, N_BUSES_PER_DIR,
        baseline_vehicles, baseline_mass / 1000.0, baseline_mass / baseline_pax)

for (label, pct) in elec_levels
    fleet = calculate_electrified_fleet(pct)
    vehicles = fleet.n_cars + fleet.n_ecars + fleet.n_suvs + fleet.n_esuvs + N_BUSES_PER_DIR
    mass = fleet.n_cars * 1500.0 + fleet.n_ecars * 1700.0 +
           fleet.n_suvs * 2200.0 + fleet.n_esuvs * 2600.0 +
           N_BUSES_PER_DIR * (12000.0 + DEFAULT_BUS_OCC * 75.0)
    pax = (fleet.n_cars + fleet.n_ecars + fleet.n_suvs + fleet.n_esuvs) * PASSENGERS_PER_CAR +
          N_BUSES_PER_DIR * DEFAULT_BUS_OCC
    @printf("  %-20s → cars=%d, ecars=%d, suvs=%d, esuvs=%d, buses=%d | vehicles=%d, mass=%.1f t, kg/pax=%.0f\n",
            label, fleet.n_cars, fleet.n_ecars, fleet.n_suvs, fleet.n_esuvs, N_BUSES_PER_DIR,
            vehicles, mass / 1000.0, mass / pax)
end

# Build full scenario list: 3 levels × 2 regen conditions = 6 scenarios
scenarios = Tuple{String, Int, Int, Int, Int, Float64}[]
for (label, pct) in elec_levels
    fleet = calculate_electrified_fleet(pct)
    push!(scenarios, ("$(label)\n(with regen)", fleet.n_cars, fleet.n_ecars, fleet.n_suvs, fleet.n_esuvs, 0.80))
    push!(scenarios, ("$(label)\n(no regen)",   fleet.n_cars, fleet.n_ecars, fleet.n_suvs, fleet.n_esuvs, 0.0))
end

# ============================================================================
# Run simulations
# ============================================================================

println("\n" * "=" ^ 70)
println("RUNNING SIMULATIONS")
println("=" ^ 70)

# Baseline (with regen) — this is the reference
println("\n--- Baseline (with regen) ---")
baseline = run_scenario_repeats("Baseline (with regen)";
    n_cars=N_CARS_PER_DIR, n_ecars=N_ECARS_PER_DIR,
    n_suvs=N_SUVS_PER_DIR, n_esuvs=N_ESUVS_PER_DIR, regen_eff=0.80)
baseline_ufp_sorted  = sort(baseline.ufp)
baseline_pm25_sorted = sort(baseline.pm25)

# Scenarios
ratio_data = DataFrame()
scenario_means = Dict{String, Tuple{Float64, Float64}}()

for (name, nc, nec, ns, nes, regen) in scenarios
    println("\n--- $name ---")
    result = run_scenario_repeats(name;
        n_cars=nc, n_ecars=nec, n_suvs=ns, n_esuvs=nes, regen_eff=regen)

    scenario_means[name] = (mean(result.ufp), mean(result.pm25))

    scenario_ufp_sorted  = sort(result.ufp)
    scenario_pm25_sorted = sort(result.pm25)

    ufp_ratio  = scenario_ufp_sorted  ./ baseline_ufp_sorted
    pm25_ratio = scenario_pm25_sorted ./ baseline_pm25_sorted

    n = length(ufp_ratio)
    append!(ratio_data, DataFrame(
        Scenario  = fill(name, n),
        UFP_ratio  = ufp_ratio,
        PM25_ratio = pm25_ratio,
    ))
end

scenario_order = [s[1] for s in scenarios]
ratio_data.Scenario = categorical(ratio_data.Scenario, levels=scenario_order, ordered=true)

# ============================================================================
# Raincloud plots
# ============================================================================

println("\n" * "=" ^ 70)
println("CREATING FIGURES")
println("=" ^ 70)

# 3 colours for 3 electrification levels, each used for with/without regen pair
colors = ["#FF8C00", "#FF8C00", "#DAA520", "#DAA520", "#2E8B57", "#2E8B57"]

# UFP ratio raincloud
println("  Creating UFP ratio raincloud...")
p_ufp = ggplot(ratio_data, @aes(x=Scenario, y=UFP_ratio, color=Scenario)) +
    geom_rainclouds(side=:right, center_boxplot=false) +
    scale_color_manual(values=colors) +
    labs(x="", y="UFP Relative to Baseline", title="(a) UFP Exposure Ratio") +
    theme_minimal() +
    guides(color="none")

ggsave(p_ufp, joinpath(FIGURES_DIR, "fig5_regen_ufp.pdf"), width=900, height=500)
ggsave(p_ufp, joinpath(FIGURES_DIR, "fig5_regen_ufp.png"), width=900, height=500)
println("  Saved fig5_regen_ufp.pdf/png")

# PM2.5 ratio raincloud
println("  Creating PM2.5 ratio raincloud...")
p_pm25 = ggplot(ratio_data, @aes(x=Scenario, y=PM25_ratio, color=Scenario)) +
    geom_rainclouds(side=:right, center_boxplot=false) +
    scale_color_manual(values=colors) +
    labs(x="", y="PM₂.₅ Relative to Baseline", title="(b) PM₂.₅ Exposure Ratio") +
    theme_minimal() +
    guides(color="none")

ggsave(p_pm25, joinpath(FIGURES_DIR, "fig5_regen_pm25.pdf"), width=900, height=500)
ggsave(p_pm25, joinpath(FIGURES_DIR, "fig5_regen_pm25.png"), width=900, height=500)
println("  Saved fig5_regen_pm25.pdf/png")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 70)
println("SUMMARY STATISTICS")
println("=" ^ 70)

@printf("  Baseline (with regen):  UFP mean=%.0f, PM2.5 mean=%.1f  [REFERENCE]\n",
        mean(baseline.ufp), mean(baseline.pm25))

for label in scenario_order
    sub = filter(r -> string(r.Scenario) == label, ratio_data)
    @printf("  %-30s  UFP ratio: median=%.3f mean=%.3f | PM2.5 ratio: median=%.3f mean=%.3f\n",
            replace(label, "\n" => " "),
            median(sub.UFP_ratio), mean(sub.UFP_ratio),
            median(sub.PM25_ratio), mean(sub.PM25_ratio))
end

# ============================================================================
# Save summary CSV (with-regen rows + baseline, for figure 8)
# ============================================================================

summary_df = DataFrame(
    scenario = String[],
    ufp_mean = Float64[],
    pm25_mean = Float64[],
    ufp_ratio_mean = Float64[],
    pm25_ratio_mean = Float64[],
)
push!(summary_df, ("Baseline", mean(baseline.ufp), mean(baseline.pm25), 1.0, 1.0))

for label in scenario_order
    ufp_m, pm25_m = scenario_means[label]
    sub = filter(r -> string(r.Scenario) == label, ratio_data)
    push!(summary_df, (replace(label, "\n" => " "), ufp_m, pm25_m,
                        mean(sub.UFP_ratio), mean(sub.PM25_ratio)))
end

summary_file = joinpath(DATA_DIR, "electrification_summary.csv")
CSV.write(summary_file, summary_df)
println("\nSaved summary: $summary_file")

println("\nFigure 5 complete!")
