# Manual Validation Template

Use this template for 3-4 representative tiles in QGIS or the Streamlit outputs.

## Tile Review Table

| Tile | Dominant Scene | Correct Classes | Main Errors | Notes |
|---|---|---|---|---|
| `tile_00001` | urban / industrial / vegetation / water | | | |
| `tile_00002` | urban / industrial / vegetation / water | | | |
| `tile_00003` | urban / industrial / vegetation / water | | | |
| `tile_00004` | urban / industrial / vegetation / water | | | |

## Suggested Error Types

- `impervious_surface` confused with `bare_soil`
- `smoke_plume` missed in low contrast scene
- `water` confused with `shadow_ignore`
- `active_fire` fragmented into small polygons

## Suggested Demo Screenshots

1. AOI boundary over source GeoTIFF
2. Raw masks for one representative tile
3. OBIA-smoothed output
4. Final GeoJSON / GeoPackage layer in QGIS

## Notes For Defense

- `confidence` is a geometric proxy, not a model posterior probability.
- AOI clipping is applied during tiling via the optional AOI vector.
- Main evaluated pipeline is `rule-based + OBIA`.
- YOLO branch is retained as a future extension, not the main validated workflow.
