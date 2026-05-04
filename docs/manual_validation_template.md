# Manual Validation Template

Manual QA is visual comparison in QGIS/Streamlit because no ground-truth labels are available.

Use this table for 3-4 representative demo territories or tiles.

## Representative Demo Review

| Tile / territory | Scene type | Good classes | Main errors | Demo quality |
|---|---|---|---|---|
| `territory_01` | dense urban / industrial / mixed vegetation | | | Strong / Medium / Weak |
| `territory_02` | suburban / bare ground / construction edge | | | Strong / Medium / Weak |
| `territory_03` | vegetation-heavy / peripheral urban | | | Strong / Medium / Weak |
| `territory_04` | industrial / roads / open soil | | | Strong / Medium / Weak |

## Suggested QA Notes

- Stable classes under review:
  - `vegetation`
  - `impervious_surface`
  - `bare_soil`
- Compare final polygons against the source GeoTIFF and AOI boundary.
- Flag where `impervious_surface` and `bare_soil` are visually close.
- Note whether the AOI clipping behaves as expected.

## Confidence Explanation

- `confidence_method = geometry_proxy`
- The score is based on class baseline, object area, and valid pixel ratio.
- It is not model posterior probability.

## Suggested Screenshots For Demo

1. Source GeoTIFF
2. Raw mask preview
3. OBIA-smoothed result
4. Final QGIS layer
5. Metrics / download panel

## Recommended Defense Notes

- Stable mode is the validated demo path.
- Experimental smoke/fire AI is optional and separated from the stable GIS export.
- Manual QA is based on visual inspection of representative territories because the provided data does not include ground-truth labels for all candidate classes.
