from pathlib import Path
import argparse
import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.ops import unary_union


FINAL_CLASSES = {
    "vegetation",
    "impervious_surface",
    "smoke_plume",
    "active_fire",
    "water",
    "bare_soil",
}


def estimate_utm_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Pick a metric UTM zone from the centroid in WGS 84."""
    geo = gdf[gdf.geometry.notnull()].copy()
    geo = geo[~geo.geometry.is_empty]

    if geo.empty:
        return CRS.from_epsg(32643)

    geo_ll = geo.to_crs("EPSG:4326")
    centroid = unary_union(geo_ll.geometry).centroid

    lon, lat = float(centroid.x), float(centroid.y)

    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)

    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(epsg)


def compactness(geom):
    area = geom.area
    perim = geom.length
    if perim == 0:
        return 0.0
    return float(4 * np.pi * area / (perim * perim))


def elongation_ratio(geom):
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)

    if len(coords) < 5:
        return 1.0

    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        edges.append(d)

    edges = sorted(edges)
    short = max(edges[0], 1e-9)
    long = edges[-1]
    return float(long / short)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_geojson")
    parser.add_argument("--output", default="outputs/geojson/almaty/Almaty_1_objects_clean.geojson")
    parser.add_argument("--metrics", default="outputs/geojson/almaty/summary_metrics.csv")
    parser.add_argument("--min-area", type=float, default=25.0)
    args = parser.parse_args()

    gdf = gpd.read_file(args.input_geojson)

    if "class" not in gdf.columns:
        raise ValueError("Expected a 'class' column")

    # keep original CRS for saving later
    original_crs = gdf.crs

    metric_crs = estimate_utm_crs(gdf)
    gdf = gdf.to_crs(metric_crs)

    gdf = gdf[gdf["class"].isin(FINAL_CLASSES)].copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["area_m2"] >= args.min_area].copy()

    gdf["compactness_score"] = gdf.geometry.apply(compactness)
    gdf["elongation_ratio"] = gdf.geometry.apply(elongation_ratio)

    gdf["subclass"] = gdf["class"]

    # Linear impervious objects are often roads / paths.
    gdf.loc[
        (gdf["class"] == "impervious_surface") & (gdf["elongation_ratio"] >= 5.0),
        "subclass"
    ] = "linear_impervious"

    # Compact impervious regions often correspond to buildings or yards.
    gdf.loc[
        (gdf["class"] == "impervious_surface")
        & (gdf["elongation_ratio"] < 2.2)
        & (gdf["compactness_score"] > 0.35),
        "subclass"
    ] = "block_like_impervious"

    # Compact vegetation may indicate landscaped patches or green roofs.
    gdf.loc[
        (gdf["class"] == "vegetation")
        & (gdf["elongation_ratio"] < 2.0)
        & (gdf["compactness_score"] > 0.45),
        "subclass"
    ] = "possible_green_roof"

    gdf.loc[
        (gdf["class"] == "water") &
        (gdf["elongation_ratio"] >= 4.0),
        "subclass"
    ] = "linear_water"

    gdf.loc[
        (gdf["class"] == "smoke_plume") &
        (gdf["compactness_score"] < 0.2),
        "subclass"
    ] = "diffuse_smoke"

    metrics = (
        gdf.groupby("class")
        .agg(
            objects=("class", "count"),
            total_area_m2=("area_m2", "sum"),
            mean_area_m2=("area_m2", "mean"),
            median_area_m2=("area_m2", "median"),
        )
        .reset_index()
    )

    total_area = metrics["total_area_m2"].sum()
    metrics["area_percent"] = metrics["total_area_m2"] / total_area * 100

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # save back in original CRS for QGIS overlay
    if original_crs is not None:
        gdf = gdf.to_crs(original_crs)

    gdf.to_file(out, driver="GeoJSON")
    metrics.to_csv(args.metrics, index=False)

    print("Saved clean GeoJSON:", out)
    print("Saved metrics CSV:", args.metrics)
    print(metrics)


if __name__ == "__main__":
    main()
