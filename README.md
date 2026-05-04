# GeoForce

GeoForce is a hackathon prototype for automated GIS layer generation from satellite imagery. It processes GeoTIFF scenes inside an optional AOI, extracts stable land-cover classes, and exports QGIS-ready GeoJSON and GeoPackage layers for downstream use in GIS and geoportals.

## Project Positioning

The stable demo path is focused on robust GIS extraction, not fine-grained roof detection.

Stable classes:

- `vegetation`
- `impervious_surface`
- `bare_soil`

Experimental classes:

- smoke / fire candidates via an optional YOLO module

Why roofs and roads are merged:

- RGB satellite imagery often makes roofs and roads visually similar, so the stable pipeline groups them as `impervious_surface` for more robust GIS extraction.

Confidence:

- `confidence_method = geometry_proxy`
- the score is based on class baseline, object area, and valid pixel ratio
- it is not model posterior probability

## Local Setup

Recommended Python: `3.11`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Stable CLI Pipeline

```bash
# 1) Tile the source GeoTIFF
python src/tiling.py path/to/image.tif --output data/tiles/my_run --aoi path/to/aoi.geojson

# 2) Create CV masks
python src/rules.py data/tiles/my_run --output outputs/masks/my_run

# 3) Smooth classes with OBIA
python src/obia.py outputs/masks/my_run/mask_index.json --output outputs/masks_obia/my_run

# 4) Polygonize masks into GIS features
python src/polygonize.py outputs/masks_obia/my_run/mask_index.json \
  --output outputs/geojson/my_run/objects.geojson

# 5) Post-process and compute summary metrics
python src/postprocess.py outputs/geojson/my_run/objects.geojson \
  --output outputs/geojson/my_run/objects_clean.geojson \
  --metrics outputs/geojson/my_run/summary_metrics.csv \
  --min-area 25

# 6) Export GeoPackage
python src/export.py outputs/geojson/my_run/objects_clean.geojson \
  --output outputs/exports/my_run/objects_clean.gpkg
```

## Streamlit Demo

Run from the repo root:

```bash
streamlit run streamlit_app.py
```

The Streamlit app supports:

- ZIP upload with GeoTIFF scenes
- local `.tif/.tiff` path input
- optional AOI path
- stable GIS extraction mode
- experimental emergency AI mode

Stable mode exports only validated land-cover classes:

- `vegetation`
- `impervious_surface`
- `bare_soil`

Experimental emergency AI is kept separate and should be treated as candidate detections only.

If optional YOLO weights are not present, the app will show a friendly message instead of crashing:

- add weights to `models/emergency.pt` to enable the experimental smoke/fire module

## Experimental Emergency AI

The optional smoke/fire module is not mixed into the stable GIS export. If enabled, it runs separately on tiles and saves outputs to:

```bash
outputs/yolo/emergency
```

These outputs are labeled as candidate detections, not validated smoke/fire classes.

## Output Layers

Typical GIS attributes include:

- `id`
- `class`
- `confidence`
- `confidence_method`
- `source`
- `tile_name`
- `area_m2`
- `valid_ratio`
- polygon geometry

The stable layers are designed to open in QGIS without manual geometry repair when the input GeoTIFF is properly georeferenced.

## Manual QA

Because the provided dataset does not include ground-truth labels for every possible class, the recommended quality review is manual visual QA on representative territories.

See:

- [docs/manual_validation_template.md](/Users/nurayzhumasseiit/hardware%20challenge/geoforce/docs/manual_validation_template.md)

Recommended screenshots:

1. source GeoTIFF
2. raw mask preview
3. OBIA-smoothed result
4. final QGIS layer
5. metrics / downloads panel

## Docker

Build:

```bash
docker build -t geoforce .
```

Run:

```bash
docker run -p 8501:8501 geoforce
```

Then open Streamlit at:

- `http://localhost:8501`

## Notes

- The main evaluated workflow is `CV masks + OBIA + polygonization + GIS export`.
- The optional YOLO emergency module is intentionally separated from the stable output.
- Do not interpret `confidence` as a neural network probability; it is a geometry-based proxy score.
