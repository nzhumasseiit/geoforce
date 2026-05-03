from pathlib import Path
import argparse

import geopandas as gpd


def export_vector(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(input_path)
    suffix = output_path.suffix.lower()

    if suffix == ".gpkg":
        gdf.to_file(output_path, driver="GPKG")
    elif suffix == ".shp":
        gdf.to_file(output_path, driver="ESRI Shapefile")
    elif suffix in {".geojson", ".json"}:
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError("Unsupported output format. Use .gpkg, .shp, or .geojson")

    print(f"Exported vector layer: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_vector", help="Path to input vector layer")
    parser.add_argument("--output", required=True, help="Output .gpkg/.shp/.geojson path")
    args = parser.parse_args()

    export_vector(args.input_vector, args.output)


if __name__ == "__main__":
    main()
