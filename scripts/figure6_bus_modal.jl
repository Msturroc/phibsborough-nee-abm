#!/usr/bin/env julia
"""
Figure 6: Passenger-Preserving Bus Modal Shift (v2)

Quantile-ratio approach: sort baseline and scenario distributions, divide
element-wise. Deviations from 1.0 show the distributional policy effect.
"""

include("common_setup.jl")
using TidierPlots
using CategoricalArrays
using StatsBase

const FIGURES_DIR = joinpath(@__DIR__, "..", "Figures")
const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(FIGURES_DIR)

println("=" ^ 70)
println("FIGURE 6: BUS MODAL SHIFT (quantile-ratio, 4-seed)")
println("=" ^ 70)

# ============================================================================
# Fleet calculator
# ============================================================================

function calculate_bus_fleet(shift_pct::Float64)
    # Work per-direction (model spawns identical fleets in each direction)
    private_per_dir = N_CARS_PER_DIR + N_ECARS_PER_DIR + N_SUVS_PER_DIR + N_ESUVS_PER_DIR
    pax_per_dir = private_per_dir * PASSENGERS_PER_CAR
    pax_to_shift = pax_per_dir * shift_pct

    # Add enough buses to keep occupancy near baseline (~22.5 pax/bus)
    extra_buses = ceil(Int, pax_to_shift / DEFAULT_BUS_OCC)
    n_b = N_BUSES_PER_DIR + extra_buses

    # Actual bus occupancy (spread all bus passengers across all buses)
    total_bus_pax = N_BUSES_PER_DIR * DEFAULT_BUS_OCC + pax_to_shift
    bus_occ = total_bus_pax / n_b

    # Remove private vehicles proportionally (preserve type ratios)
    private_removed = round(Int, pax_to_shift / PASSENGERS_PER_CAR)
    new_private = private_per_dir - private_removed

    car_frac  = N_CARS_PER_DIR  / private_per_dir
    ecar_frac = N_ECARS_PER_DIR / private_per_dir
    suv_frac  = N_SUVS_PER_DIR  / private_per_dir
    esuv_frac = N_ESUVS_PER_DIR / private_per_dir

    n_c  = round(Int, new_private * car_frac)
    n_ec = round(Int, new_private * ecar_frac)
    n_s  = round(Int, new_private * suv_frac)
    n_es = round(Int, new_private * esuv_frac)

    return (n_c=n_c, n_ec=n_ec, n_s=n_s, n_es=n_es, n_b=n_b, bus_occ=bus_occ)
end

# ============================================================================
# Scenario definitions
# ============================================================================

scenarios = [
    ("25% to Bus", 0.25),
    ("50% to Bus", 0.50),
    ("75% to Bus", 0.75),
]

# Print fleet configurations
println("\nFLEET CONFIGURATIONS (per direction)")
println("=" ^ 70)
println("  Baseline             → cars=$(N_CARS_PER_DIR), ecars=$(N_ECARS_PER_DIR), suvs=$(N_SUVS_PER_DIR), esuvs=$(N_ESUVS_PER_DIR), buses=$(N_BUSES_PER_DIR)")
for (name, shift) in scenarios
    fleet = calculate_bus_fleet(shift)
    bus_weight = 12000.0 + fleet.bus_occ * 75.0
    @printf("  %-20s → cars=%d, ecars=%d, suvs=%d, esuvs=%d, buses=%d, occ=%.1f, bus_wt=%.0f kg\n",
            name, fleet.n_c, fleet.n_ec, fleet.n_s, fleet.n_es, fleet.n_b, fleet.bus_occ, bus_weight)
end

# ============================================================================
# Run simulations
# ============================================================================

println("\n" * "=" ^ 70)
println("RUNNING SIMULATIONS")
println("=" ^ 70)

# Baseline
println("\n--- Baseline ---")
baseline = run_scenario_repeats("Baseline";
    n_cars=N_CARS_PER_DIR, n_ecars=N_ECARS_PER_DIR,
    n_suvs=N_SUVS_PER_DIR, n_esuvs=N_ESUVS_PER_DIR,
    n_buses=N_BUSES_PER_DIR, regen_eff=0.80)
baseline_ufp_sorted  = sort(baseline.ufp)
baseline_pm25_sorted = sort(baseline.pm25)

# Scenarios
ratio_data = DataFrame()
scenario_means = Dict{String, Tuple{Float64, Float64}}()

for (name, shift) in scenarios
    fleet = calculate_bus_fleet(shift)
    println("\n--- $name ---")
    result = run_scenario_repeats(name;
        n_cars=fleet.n_c, n_ecars=fleet.n_ec,
        n_suvs=fleet.n_s, n_esuvs=fleet.n_es,
        n_buses=fleet.n_b, bus_occ=fleet.bus_occ,
        regen_eff=0.80)

    scenario_means[name] = (mean(result.ufp), mean(result.pm25))

    scenario_ufp_sorted  = sort(result.ufp)
    scenario_pm25_sorted = sort(result.pm25)

    ufp_ratio  = scenario_ufp_sorted  ./ baseline_ufp_sorted
    pm25_ratio = scenario_pm25_sorted ./ baseline_pm25_sorted

    n = length(ufp_ratio)
    append!(ratio_data, DataFrame(
        Scenario   = fill(name, n),
        UFP_ratio  = ufp_ratio,
        PM25_ratio = pm25_ratio,
    ))
end

scenario_order = first.(scenarios)
ratio_data.Scenario = categorical(ratio_data.Scenario, levels=scenario_order, ordered=true)

# ============================================================================
# Raincloud plots
# ============================================================================

println("\n" * "=" ^ 70)
println("CREATING FIGURES")
println("=" ^ 70)

scenario_colors = ["#FF8C00", "#DAA520", "#2E8B57"]

# UFP
println("Generating UFP ratio raincloud...")
p_ufp = ggplot(ratio_data, @aes(x=Scenario, y=UFP_ratio, color=Scenario)) +
    geom_rainclouds(side=:right, center_boxplot=false) +
    scale_color_manual(values=scenario_colors) +
    labs(x="", y="UFP Relative to Baseline",
         title="Passenger-Preserving Bus Modal Shift") +
    theme_minimal() +
    guides(color="none")

ggsave(p_ufp, joinpath(FIGURES_DIR, "fig6_bus_modal_ufp.pdf"), width=600, height=450)
ggsave(p_ufp, joinpath(FIGURES_DIR, "fig6_bus_modal_ufp.png"), width=600, height=450)
println("  Saved: Figures/fig6_bus_modal_ufp.pdf")

# PM2.5
println("Generating PM2.5 ratio raincloud...")
p_pm25 = ggplot(ratio_data, @aes(x=Scenario, y=PM25_ratio, color=Scenario)) +
    geom_rainclouds(side=:right, center_boxplot=false) +
    scale_color_manual(values=scenario_colors) +
    labs(x="", y="PM₂.₅ Relative to Baseline",
         title="Passenger-Preserving Bus Modal Shift") +
    theme_minimal() +
    guides(color="none")

ggsave(p_pm25, joinpath(FIGURES_DIR, "fig6_bus_modal_pm25.pdf"), width=600, height=450)
ggsave(p_pm25, joinpath(FIGURES_DIR, "fig6_bus_modal_pm25.png"), width=600, height=450)
println("  Saved: Figures/fig6_bus_modal_pm25.pdf")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 70)
println("SUMMARY STATISTICS")
println("=" ^ 70)

@printf("  Baseline:  UFP mean=%.0f, PM2.5 mean=%.1f  [REFERENCE]\n",
        mean(baseline.ufp), mean(baseline.pm25))

for label in scenario_order
    sub = filter(r -> string(r.Scenario) == label, ratio_data)
    @printf("  %-20s  UFP ratio: median=%.3f mean=%.3f | PM2.5 ratio: median=%.3f mean=%.3f\n",
            label,
            median(sub.UFP_ratio), mean(sub.UFP_ratio),
            median(sub.PM25_ratio), mean(sub.PM25_ratio))
end

# Save summary data for figure8 (policy comparison)
summary_file = joinpath(DATA_DIR, "bus_modal_summary.csv")
mkpath(DATA_DIR)
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
    push!(summary_df, (label, ufp_m, pm25_m, mean(sub.UFP_ratio), mean(sub.PM25_ratio)))
end
CSV.write(summary_file, summary_df)
println("\nSaved summary: $summary_file")

println("\nFigure 6 complete!")
