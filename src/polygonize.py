from pathlib import Path
import argparse
import json
import uuid

import cv2
import numpy as np
import geopandas as gpd
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from affine import Affine
from pyproj import CRS


CLASS_CONFIG = {
    "vegetation": {
        "min_area_m2": 10,
        "simplify_tolerance": 0,
        "confidence_base": 62,
    },
    "impervious_surface": {
        "min_area_m2": 80,
        "simplify_tolerance": 0,
        "confidence_base": 58,
    },
    "bare_soil": {
        "min_area_m2": 30,
        "simplify_tolerance": 0,
        "confidence_base": 50,
    },
    "shadow_ignore": {
        "min_area_m2": 100,
        "simplify_tolerance": 0,
        "confidence_base": 40,
    },
}


def load_mask(mask_path: str) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 0


def affine_from_list(values) -> Affine:
    # Saved as first six affine values: a, b, c, d, e, f
    return Affine(*values)


def pixel_to_geo(transform: Affine, x: float, y: float):
    lon, lat = transform * (x, y)
    return lon, lat


def mask_to_polygons(mask: np.ndarray, transform: Affine, simplify_tolerance: float):
    mask_u8 = (mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:
        if len(contour) < 4:
            continue

        points = contour[:, 0, :]

        geo_points = [
            pixel_to_geo(transform, float(x), float(y))
            for x, y in points
        ]

        if len(geo_points) < 4:
            continue

        poly = Polygon(geo_points)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            continue

        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if not poly.is_empty:
            polygons.append(poly)

    return polygons


def estimate_utm_crs(gdf: gpd.GeoDataFrame):
    geo = gdf[gdf.geometry.notnull()].copy()
    geo = geo[~geo.geometry.is_empty]

    if geo.empty:
        return CRS.from_epsg(32643)

    if geo.crs is None:
        return CRS.from_epsg(32643)

    geo_ll = geo.to_crs("EPSG:4326")
    centroid = unary_union(geo_ll.geometry).centroid
    lon, lat = centroid.x, centroid.y

    zone = int((lon + 180) // 6) + 1

    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(epsg)


def calculate_area_m2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["area_m2"] = []
        return gdf

    metric_crs = estimate_utm_crs(gdf)
    gdf_metric = gdf.to_crs(metric_crs)
    gdf["area_m2"] = gdf_metric.geometry.area.round(2)

    return gdf


def confidence_from_geometry(class_name: str, area_m2: float, valid_ratio: float):
    cfg = CLASS_CONFIG.get(class_name, {})
    base = cfg.get("confidence_base", 60)

    area_score = min(area_m2 / 200.0, 1.0) * 15
    valid_score = min(valid_ratio, 1.0) * 10

    confidence = base + area_score + valid_score
    return round(float(min(confidence, 98)), 2)


def polygonize_masks(mask_index_path: str, output_path: str, merge_same_class: bool = False):
    mask_index_path = Path(mask_index_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mask_index_path, "r", encoding="utf-8") as f:
        mask_records = json.load(f)

    features = []

    for record in mask_records:
        class_name = record["class"]

        if class_name not in CLASS_CONFIG:
            continue

        mask_path = record["mask_path"]
        meta_path = record["metadata_path"]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        crs = meta["crs"]
        transform = affine_from_list(meta["transform"])
        valid_ratio = float(meta.get("valid_ratio", 1.0))

        cfg = CLASS_CONFIG[class_name]

        mask = load_mask(mask_path)

        polygons = mask_to_polygons(
            mask=mask,
            transform=transform,
            simplify_tolerance=cfg["simplify_tolerance"],
        )

        for poly in polygons:
            features.append({
                "id": f"{class_name}_{uuid.uuid4().hex[:10]}",
                "class": class_name,
                "is_final_class": class_name != "shadow_ignore",
                "source": meta["source"],
                "tile_name": meta["tile_name"],
                "valid_ratio": valid_ratio,
                "aoi_path": meta.get("aoi_path"),
                "geometry": poly,
            })

    if not features:
        print("No polygons created.")
        return None

    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs=crs)

    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = calculate_area_m2(gdf)

    keep_rows = []

    for idx, row in gdf.iterrows():
        cfg = CLASS_CONFIG[row["class"]]
        if row["area_m2"] >= cfg["min_area_m2"]:
            keep_rows.append(idx)

    gdf = gdf.loc[keep_rows].copy()

    if gdf.empty:
        print("All polygons filtered out by area thresholds.")
        return None

    gdf["confidence"] = gdf.apply(
        lambda row: confidence_from_geometry(
            row["class"],
            row["area_m2"],
            row["valid_ratio"],
        ),
        axis=1,
    )
    gdf["confidence_method"] = "geometry_proxy"

    if merge_same_class:
        merged = []

        for class_name, group in gdf.groupby("class"):
            geom = unary_union(group.geometry)

            if isinstance(geom, Polygon):
                geoms = [geom]
            elif isinstance(geom, MultiPolygon):
                geoms = list(geom.geoms)
            else:
                continue

            for geom_part in geoms:
                merged.append({
                    "id": f"{class_name}_{uuid.uuid4().hex[:10]}",
                    "class": class_name,
                    "source": ",".join(sorted(group["source"].unique())),
                    "tile_name": "merged",
                    "valid_ratio": round(float(group["valid_ratio"].mean()), 4),
                    "confidence": round(float(group["confidence"].mean()), 2),
                    "confidence_method": "geometry_proxy",
                    "geometry": geom_part,
                })

        gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)
        gdf = calculate_area_m2(gdf)

    # GeoJSON likes EPSG:4326.
    gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[
        [
            "id",
            "class",
            "is_final_class",
            "confidence",
            "confidence_method",
            "source",
            "tile_name",
            "area_m2",
            "valid_ratio",
            "aoi_path",
            "geometry",
        ]
    ]

    gdf.to_file(output_path, driver="GeoJSON")

    print("=" * 60)
    print(f"Saved GeoJSON: {output_path}")
    print(f"Objects: {len(gdf)}")
    print("Classes:")
    print(gdf["class"].value_counts())
    print("=" * 60)

    return gdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_index", help="Path to mask_index.json from obia.py")
    parser.add_argument("--output", default="outputs/geojson/objects.geojson")
    parser.add_argument("--merge-same-class", action="store_true")
    args = parser.parse_args()

    polygonize_masks(
        mask_index_path=args.mask_index,
        output_path=args.output,
        merge_same_class=args.merge_same_class,
    )


if __name__ == "__main__":
    main()
