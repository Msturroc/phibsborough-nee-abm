# Non-Exhaust Emissions Agent-Based Model

The Phibsborough junction in Dublin is a documented non-exhaust emission hotspot. Street-level measurements from [Google Air View](https://www.google.com/intl/en_ie/about/stories/clean-air/) show ultrafine particle concentrations 70-80% above the Dublin average, largely due to heavy braking at the traffic-light-controlled intersection. With exhaust emissions declining through electrification, brake and tyre wear are becoming the dominant source of traffic-related particulate matter in cities.

This repository contains an agent-based model that simulates the junction at high spatial resolution (200m x 200m grid, adaptive sub-second timestep). Cars, SUVs, electric variants, buses, and microcars navigate the intersection, generating mass-dependent brake and tyre wear emissions that disperse and decay across the grid. A moving observer vehicle samples emissions the same way as the Google Air View car, so simulated and measured distributions can be compared directly.

The model is calibrated against the Air View data using CMA-ES (15 parameters, 2000 evaluations). Global sensitivity analysis via Polynomial Chaos Expansion shows that fleet composition, mean speed, and speed variability interact strongly with each other (all first-order Sobol indices < 0.013, but total-order indices 0.23-0.34). Scenario scripts cover speed limits, electrification with and without regenerative braking, bus modal shift, microcar adoption, and combined policies.

![Simulation Animation](Figures/readme_animation.gif)

## Getting Started

Requires Julia >= 1.10. Install dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The calibration data (`AirView_DublinCity_Measurements_ugm3.csv`, 581 MB) is not included due to size. Obtain it from the [Google Environmental Insights Explorer](https://insights.sustainability.google/) and place it in the repository root.

Calibrated parameters are already provided in `data/`. To re-run calibration:

```bash
julia src/cmaes_ecs_v7.jl
```

To reproduce figures:

```bash
julia scripts/figure2_model.jl
julia scripts/figure3_gsa.jl
julia scripts/figure4_speed_var.jl
# etc.
```

To run the GSA with distributed parallelism:

```bash
julia -p 30 src/gsa_passenger_preserving_v5.jl
```
