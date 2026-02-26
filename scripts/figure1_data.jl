#!/usr/bin/env julia
# Figure 1: Study Area and Experimental Data
# Panel (a): Map of Dublin with 200m × 200m study area box
# Panel (b): PM2.5 spatial heatmap from Google Air View
# Panel (c): UFP spatial heatmap from Google Air View

using CSV
using DataFrames
using Statistics
using Plots
using StatsBase
using Images
using FileIO

println("="^60)
println("Figure 1: Study Area and Experimental Data")
println("="^60)

# ============================================================================
# Load and process the Dublin map image
# ============================================================================

println("\nLoading Dublin map with study area...")
map_img = load(joinpath(@__DIR__, "..", "data", "box_200x200.png"))
println("  Map dimensions: $(size(map_img))")

# ============================================================================
# Load Google Air View data
# ============================================================================

println("\nLoading Google Air View data...")
full_airview_data = CSV.read(joinpath(@__DIR__, "..", "AirView_DublinCity_Measurements_ugm3.csv"), DataFrame)

# 200m × 200m box centered on Phibsborough junction
lat_min, lat_max = 53.359899, 53.361701
lon_min, lon_max = -6.27421, -6.27119

# Filter to box
phibs_data = filter(row -> (lat_min <= row.latitude <= lat_max) &&
                           (lon_min <= row.longitude <= lon_max), full_airview_data)

println("  Data points in study area: $(nrow(phibs_data))")

# Extract PM2.5 data
pm25_data = dropmissing(phibs_data, :PM25_ugm3)
pm25_lats = pm25_data.latitude
pm25_lons = pm25_data.longitude
pm25_vals = Float64.(pm25_data.PM25_ugm3)

# Extract UFP data (PMch1 + PMch2)
ufp_data = dropmissing(phibs_data, [:PMch1_perL, :PMch2_perL])
ufp_lats = ufp_data.latitude
ufp_lons = ufp_data.longitude
ufp_vals = Float64.(ufp_data.PMch1_perL .+ ufp_data.PMch2_perL)

println("  PM2.5 measurements: $(length(pm25_vals))")
println("  UFP measurements: $(length(ufp_vals))")

# ============================================================================
# Create spatial bins for heatmaps
# ============================================================================

n_bins = 200  # 200×200 grid (~1m resolution for 200m box)
lat_edges = range(lat_min, lat_max, length=n_bins+1)
lon_edges = range(lon_min, lon_max, length=n_bins+1)

function bin_spatial_data(lats, lons, vals, lat_edges, lon_edges)
    n_lat = length(lat_edges) - 1
    n_lon = length(lon_edges) - 1

    sum_grid = zeros(n_lat, n_lon)
    count_grid = zeros(Int, n_lat, n_lon)

    for i in 1:length(vals)
        lat_idx = searchsortedlast(collect(lat_edges), lats[i])
        lon_idx = searchsortedlast(collect(lon_edges), lons[i])

        if 1 <= lat_idx <= n_lat && 1 <= lon_idx <= n_lon
            sum_grid[lat_idx, lon_idx] += vals[i]
            count_grid[lat_idx, lon_idx] += 1
        end
    end

    mean_grid = similar(sum_grid)
    for i in eachindex(sum_grid)
        mean_grid[i] = count_grid[i] > 0 ? sum_grid[i] / count_grid[i] : NaN
    end

    return mean_grid, count_grid
end

pm25_mean, _ = bin_spatial_data(pm25_lats, pm25_lons, pm25_vals, lat_edges, lon_edges)
ufp_mean, _ = bin_spatial_data(ufp_lats, ufp_lons, ufp_vals, lat_edges, lon_edges)

# Convert to meters from box corner
lat_to_m = 111000.0
lon_to_m = 111000.0 * cosd(53.36)

lat_centers = [(lat_edges[i] + lat_edges[i+1])/2 for i in 1:n_bins]
lon_centers = [(lon_edges[i] + lon_edges[i+1])/2 for i in 1:n_bins]

x_meters = (collect(lon_centers) .- lon_min) .* lon_to_m
y_meters = (collect(lat_centers) .- lat_min) .* lat_to_m

# ============================================================================
# Create the figure
# ============================================================================

println("\nGenerating Figure 1...")

# Plot defaults
default(
    titlefontsize=12,
    guidefontsize=10,
    tickfontsize=9,
    legendfontsize=9,
    linewidth=1.5
)

# Panel (a): Dublin map with study area
p1 = plot(map_img, axis=false, ticks=false, border=:none,
    title="(a) Study Area",
    titlefontsize=11,
    aspect_ratio=:equal)

# Panel (b): PM2.5 heatmap
pm25_clim_max = quantile(filter(!isnan, vec(pm25_mean)), 0.95)
p2 = heatmap(x_meters, y_meters, pm25_mean,
    xlabel="East-West (m)",
    ylabel="North-South (m)",
    title="(b) PM₂.₅ Concentration",
    titlefontsize=11,
    color=:YlOrRd,
    colorbar_title="μg/m³",
    clims=(0, pm25_clim_max),
    framestyle=:box,
    xlims=(0, 200),
    ylims=(0, 200),
    aspect_ratio=:equal
)

# Panel (c): UFP heatmap
ufp_clim_max = quantile(filter(!isnan, vec(ufp_mean)), 0.95)
p3 = heatmap(x_meters, y_meters, ufp_mean,
    xlabel="East-West (m)",
    ylabel="North-South (m)",
    title="(c) UFP Concentration",
    titlefontsize=11,
    color=:YlOrRd,
    colorbar_title="#/L",
    clims=(0, ufp_clim_max),
    framestyle=:box,
    xlims=(0, 200),
    ylims=(0, 200),
    aspect_ratio=:equal
)

# Combine into 1×3 layout
fig = plot(p1, p2, p3,
    layout=(1, 3),
    size=(1500, 450),
    dpi=300,
    left_margin=3Plots.mm,
    right_margin=3Plots.mm,
    bottom_margin=5Plots.mm,
    top_margin=3Plots.mm
)

# ============================================================================
# Save outputs
# ============================================================================

out_dir = joinpath(@__DIR__, "..", "Figures")
isdir(out_dir) || mkpath(out_dir)

pdf_path = joinpath(out_dir, "figure1_data.pdf")
png_path = joinpath(out_dir, "figure1_data.png")

savefig(fig, pdf_path)
savefig(fig, png_path)

println("\n" * "="^60)
println("Figure 1 saved:")
println("  PDF: $pdf_path")
println("  PNG: $png_path")
println("="^60)

# Print summary statistics
println("\nData Summary:")
println("  PM2.5: mean=$(round(mean(pm25_vals), digits=1)) μg/m³, " *
        "range=$(round(minimum(pm25_vals), digits=1))-$(round(maximum(pm25_vals), digits=1)) μg/m³")
println("  UFP:   mean=$(round(mean(ufp_vals), digits=0)) #/L, " *
        "range=$(round(minimum(ufp_vals), digits=0))-$(round(maximum(ufp_vals), digits=0)) #/L")
