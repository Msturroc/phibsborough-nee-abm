#!/usr/bin/env julia
"""
Figure 3: Global Sensitivity Analysis (GSA) Results
1×3 panel: (a) S1 dot-whisker, (b) ST dot-whisker, (c) S2 heatmap
"""

using CSV, DataFrames, StatsPlots, Measures, Printf
gr()

const ROOT = joinpath(@__DIR__, "..")
const OUT_DIR = joinpath(ROOT, "Figures")

const SNAMES = Dict(
    "Mean Speed"=>"Mean Speed", "Speed Variability"=>"Speed Variability",
    "Cars"=>"Cars", "E-Cars"=>"E-Cars",
    "SUVs"=>"SUVs", "E-SUVs"=>"E-SUVs",
    "Buses"=>"Buses", "Microcars"=>"Microcars",
    "Green Duration"=>"Green Duration", "Sight Distance"=>"Sight Distance",
    "Planning Distance"=>"Planning Distance"
)

println("="^60)
println("Figure 3: GSA Results")
println("="^60)

df = CSV.read(joinpath(ROOT, "data", "gsa_δ_total_nee_indices.csv"), DataFrame)
s2_df = CSV.read(joinpath(ROOT, "data", "gsa_δ_total_nee_s2_matrix.csv"), DataFrame)

# Sort by ST descending
idx = sortperm(df.ST_mean, rev=true)
sparams = df.parameter[idx]
sshort = [SNAMES[p] for p in sparams]
n = length(sparams)

# Reversed so highest appears at top of horizontal plot
dshort = reverse(sshort)
dS1 = reverse(df.S1_mean[idx])
dS1e = reverse(df.S1_std[idx])
dST = reverse(df.ST_mean[idx])
dSTe = reverse(df.ST_std[idx])

# S2 matrix reordered to match ST sort
s2_names = String.(names(s2_df)[2:end])
S2 = Matrix{Float64}(s2_df[:, 2:end])
ri = [findfirst(==(p), s2_names) for p in sparams]
S2s = S2[ri, ri]

# Lower triangle only
S2lo = copy(S2s)
for i in 1:n, j in i:n
    S2lo[i, j] = NaN
end

# Top 3 interactions
ints = [(i, j, S2s[i,j]) for i in 2:n for j in 1:(i-1)]
sort!(ints, by=x->x[3], rev=true)
top3 = ints[1:3]

println("\nTop 3 S2 interactions:")
for (i, j, v) in top3
    @printf("  %s x %s = %.4f\n", sshort[j], sshort[i], v)
end

bc = RGB(0.27, 0.51, 0.71)

# Panel (a): S1 dot-whisker
p1 = scatter(dS1, 1:n, xerr=dS1e,
    yticks=(1:n, dshort),
    xlabel="S1 Index", title="(a) First-Order (S1)",
    titlefontsize=12, guidefontsize=10, tickfontsize=9,
    legend=false, color=bc, markerstrokecolor=bc,
    markersize=6, markershape=:circle,
    xlims=(-0.005, 0.035),
    grid=false,
    left_margin=8mm, bottom_margin=5mm, top_margin=3mm, right_margin=3mm)

# Panel (b): ST dot-whisker
p2 = scatter(dST, 1:n, xerr=dSTe,
    yticks=(1:n, dshort),
    xlabel="ST Index", title="(b) Total-Order (ST)",
    titlefontsize=12, guidefontsize=10, tickfontsize=9,
    legend=false, color=bc, markerstrokecolor=bc,
    markersize=6, markershape=:circle,
    xlims=(-0.02, 0.60),
    grid=false,
    left_margin=8mm, bottom_margin=5mm, top_margin=3mm, right_margin=3mm)

# Panel (c): S2 heatmap with numeric indices for proper centering
cmax = maximum(filter(!isnan, S2lo))
p3 = heatmap(1:n, 1:n, S2lo, c=:YlOrRd, clims=(0.0, cmax),
    colorbar_title="S2", title="(c) Pairwise Interactions (S2)",
    titlefontsize=12, guidefontsize=10, tickfontsize=8,
    xticks=(1:n, sshort), yticks=(1:n, sshort),
    yflip=true, grid=false,
    left_margin=3mm, bottom_margin=8mm,
    top_margin=3mm, right_margin=5mm,
    xrotation=45)

fig = plot(p1, p2, p3, layout=(1,3), size=(2200, 550), dpi=300,
    left_margin=5mm, right_margin=5mm, top_margin=5mm, bottom_margin=20mm)

isdir(OUT_DIR) || mkpath(OUT_DIR)
savefig(fig, joinpath(OUT_DIR, "figure3_gsa.pdf"))
savefig(fig, joinpath(OUT_DIR, "figure3_gsa.png"))

println("\nFigure 3 saved to Figures/figure3_gsa.{pdf,png}")
