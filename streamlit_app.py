import shutil
import subprocess
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st


st.set_page_config(page_title="GeoForce: Satellite Imagery to GIS Layers", layout="wide")
st.title("GeoForce: Satellite Imagery to GIS Layers")
st.caption("AOI-aware satellite image processing with GIS-ready outputs for QGIS and geospatial portals.")

STABLE_CLASSES = {"vegetation", "impervious_surface", "bare_soil"}
EXPERIMENTAL_CLASSES = {"smoke_plume", "active_fire", "water"}


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error("Command failed:")
        st.code(" ".join(cmd))
        st.error("Error:")
        st.code(result.stderr or result.stdout)
        st.stop()
    return result


def reset_pipeline_dirs(paths):
    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def prepare_input_tif(source_mode: str) -> Path | None:
    if source_mode == "ZIP upload":
        uploaded = st.sidebar.file_uploader("Input GeoTIFF ZIP", type=["zip"])
        if not uploaded:
            return None

        raw_dir = Path("data/raw/streamlit")
        reset_pipeline_dirs([raw_dir])
        raw_dir.mkdir(parents=True, exist_ok=True)

        zip_path = raw_dir / "uploaded.zip"
        zip_path.write_bytes(uploaded.read())

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(raw_dir)
        except zipfile.BadZipFile:
            st.error("Invalid ZIP file.")
            st.stop()

        tif_files = list(raw_dir.rglob("*.tif")) + list(raw_dir.rglob("*.tiff"))
        if not tif_files:
            st.error("No .tif or .tiff found inside ZIP.")
            st.stop()

        return tif_files[0]

    local_path = st.sidebar.text_input(
        "Input GeoTIFF",
        placeholder="/absolute/path/to/image.tif",
    ).strip()

    if not local_path:
        return None

    tif_path = Path(local_path).expanduser()
    if not tif_path.exists():
        st.error(f"Path does not exist: {tif_path}")
        st.stop()

    if tif_path.is_dir():
        tif_files = list(tif_path.rglob("*.tif")) + list(tif_path.rglob("*.tiff"))
        if not tif_files:
            st.error("No .tif or .tiff found in the provided folder.")
            st.stop()
        return tif_files[0]

    if tif_path.suffix.lower() not in {".tif", ".tiff"}:
        st.error("Please provide a .tif/.tiff file or a folder containing GeoTIFFs.")
        st.stop()

    return tif_path


def run_pipeline(tif_path: Path, aoi_path: str, min_area: float):
    tiles_dir = Path("data/tiles/streamlit")
    masks_dir = Path("outputs/masks/streamlit")
    masks_obia_dir = Path("outputs/masks_obia/streamlit")
    out_geo_dir = Path("outputs/geojson/streamlit")
    exports_dir = Path("outputs/exports/streamlit")

    reset_pipeline_dirs([
        tiles_dir,
        masks_dir,
        masks_obia_dir,
        out_geo_dir,
        exports_dir,
    ])

    with st.spinner("Running tiling..."):
        cmd = ["python", "src/tiling.py", str(tif_path), "--output", str(tiles_dir)]
        if aoi_path:
            cmd.extend(["--aoi", aoi_path])
        run_cmd(cmd)

    masks_dir.mkdir(parents=True, exist_ok=True)
    with st.spinner("Running rule masks..."):
        run_cmd([
            "python",
            "src/rules.py",
            str(tiles_dir),
            "--output",
            str(masks_dir),
        ])

    with st.spinner("Running OBIA..."):
        run_cmd([
            "python",
            "src/obia.py",
            str(masks_dir / "mask_index.json"),
            "--output",
            str(masks_obia_dir),
        ])

    out_geo_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    objects_path = out_geo_dir / "objects.geojson"
    objects_clean_path = out_geo_dir / "objects_clean.geojson"
    summary_csv = out_geo_dir / "summary_metrics.csv"
    objects_gpkg = exports_dir / "objects_clean.gpkg"

    with st.spinner("Polygonizing results..."):
        run_cmd([
            "python",
            "src/polygonize.py",
            str(masks_obia_dir / "mask_index.json"),
            "--output",
            str(objects_path),
        ])

    with st.spinner("Post-processing & metrics..."):
        run_cmd([
            "python",
            "src/postprocess.py",
            str(objects_path),
            "--output",
            str(objects_clean_path),
            "--metrics",
            str(summary_csv),
            "--min-area",
            str(min_area),
        ])

    with st.spinner("Exporting GeoPackage..."):
        run_cmd([
            "python",
            "src/export.py",
            str(objects_clean_path),
            "--output",
            str(objects_gpkg),
        ])

    return {
        "tif_path": str(tif_path),
        "aoi_path": aoi_path or None,
        "objects_path": str(objects_path),
        "objects_clean_path": str(objects_clean_path),
        "summary_csv": str(summary_csv),
        "objects_gpkg": str(objects_gpkg),
        "mask_index": str(masks_dir / "mask_index.json"),
        "obia_index": str(masks_obia_dir / "mask_index.json"),
    }


def load_results():
    last_run = st.session_state.get("last_run")
    if not last_run:
        return None

    objects_clean = Path(last_run["objects_clean_path"])
    summary_csv = Path(last_run["summary_csv"])

    if not objects_clean.exists():
        return None

    gdf = gpd.read_file(objects_clean)
    metrics_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()

    return {
        "meta": last_run,
        "gdf": gdf,
        "metrics_df": metrics_df,
    }


def filter_for_mode(gdf, metrics_df, detection_mode: str):
    if detection_mode == "Experimental":
        return gdf, metrics_df

    filtered_gdf = gdf[gdf["class"].isin(STABLE_CLASSES)].copy() if "class" in gdf.columns else gdf.copy()
    if metrics_df.empty or "class" not in metrics_df.columns:
        return filtered_gdf, metrics_df

    filtered_metrics = metrics_df[metrics_df["class"].isin(STABLE_CLASSES)].copy()
    return filtered_gdf, filtered_metrics


def render_kpis(gdf, metrics_df):
    objects_detected = len(gdf)
    total_area = float(gdf["area_m2"].sum()) if "area_m2" in gdf.columns else 0.0
    classes_found = int(gdf["class"].nunique()) if "class" in gdf.columns else 0
    export_formats = 3

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects detected", f"{objects_detected}")
    c2.metric("Total mapped area", f"{total_area:,.0f}")
    c3.metric("Classes found", f"{classes_found}")
    c4.metric("Export formats", f"{export_formats}")


with st.sidebar:
    st.header("Settings")
    source_mode = st.radio("Input mode", ("Local path", "ZIP upload"))
    tif_path = prepare_input_tif(source_mode)
    aoi_path = st.text_input(
        "AOI file optional",
        placeholder="/absolute/path/to/aoi.geojson",
    ).strip()
    detection_mode = st.selectbox("Detection mode", ["Stable", "Experimental"])
    min_area = st.number_input("Min object area", min_value=1.0, value=25.0, step=25.0)
    run_clicked = st.button("Run pipeline", type="primary", disabled=tif_path is None)

if tif_path is not None:
    st.success(f"Input ready: {tif_path}")
else:
    st.info("Choose a GeoTIFF source in the sidebar to begin.")

st.info(
    "Stable classes: vegetation, impervious surface, bare soil.\n\n"
    "Experimental classes: smoke/fire candidates."
)

if run_clicked and tif_path is not None:
    st.session_state["last_run"] = run_pipeline(tif_path, aoi_path, min_area)
    st.success("Pipeline finished!")

results = load_results()
active_gdf = pd.DataFrame()
active_metrics = pd.DataFrame()
if results is not None:
    active_gdf, active_metrics = filter_for_mode(results["gdf"], results["metrics_df"], detection_mode)

tabs = st.tabs(["Overview", "Results", "Metrics", "Downloads", "Experimental Emergency AI"])

with tabs[0]:
    st.subheader("Overview")
    if results is None:
        st.write("Run the pipeline to see an overview of extracted GIS layers and summary indicators.")
    else:
        render_kpis(active_gdf, active_metrics)
        st.write(
            "Stable mode focuses on vegetation, impervious surface, and bare soil. "
            "Experimental mode also exposes smoke/fire candidate classes."
        )
        st.write("Current run")
        st.json(
            {
                "input_tif": results["meta"]["tif_path"],
                "aoi": results["meta"]["aoi_path"],
                "detection_mode": detection_mode,
                "mask_index_exists": Path(results["meta"]["mask_index"]).exists(),
                "obia_index_exists": Path(results["meta"]["obia_index"]).exists(),
            }
        )

with tabs[1]:
    st.subheader("Results")
    if results is None:
        st.write("No results yet.")
    else:
        gdf = active_gdf
        preview_cols = [c for c in ["id", "class", "confidence", "confidence_method", "area_m2", "tile_name"] if c in gdf.columns]
        st.write("Detected objects preview")
        st.dataframe(gdf[preview_cols].head(50), use_container_width=True)
        st.write("Classes detected")
        st.dataframe(gdf["class"].value_counts().rename_axis("class").reset_index(name="objects"), use_container_width=True)

with tabs[2]:
    st.subheader("Metrics")
    if results is None:
        st.write("No metrics yet.")
    else:
        if active_metrics.empty:
            st.write("Metrics table not found.")
        else:
            st.dataframe(active_metrics, use_container_width=True)

with tabs[3]:
    st.subheader("Downloads")
    if results is None:
        st.write("No downloadable outputs yet.")
    else:
        objects_clean_path = Path(results["meta"]["objects_clean_path"])
        summary_csv = Path(results["meta"]["summary_csv"])
        active_geojson = active_gdf.to_json()

        if objects_clean_path.exists() and not active_gdf.empty:
            st.download_button(
                "Download GeoJSON",
                active_geojson.encode("utf-8"),
                file_name="detected_objects.geojson",
                mime="application/geo+json",
            )
        if summary_csv.exists() and not active_metrics.empty:
            st.download_button(
                "Download Metrics CSV",
                active_metrics.to_csv(index=False).encode("utf-8"),
                file_name="summary_metrics.csv",
                mime="text/csv",
            )
        if Path(results["meta"]["objects_gpkg"]).exists():
            st.caption("GeoPackage download always contains the full GIS output from the last run.")
            st.download_button(
                "Download GeoPackage",
                Path(results["meta"]["objects_gpkg"]).read_bytes(),
                file_name="detected_objects.gpkg",
                mime="application/geopackage+sqlite3",
            )

with tabs[4]:
    st.subheader("Experimental Emergency AI")
    st.write(
        "This prototype keeps the emergency-oriented classes as experimental. "
        "Use them as candidate signals rather than fully validated operational outputs."
    )
    st.write(
        "- Stable focus: vegetation, impervious surface, bare soil\n"
        "- Experimental signals: smoke plume, active fire, water-like regions\n"
        "- Recommended demo framing: AOI-aware GIS extraction with extensible emergency analytics"
    )
    if detection_mode == "Experimental":
        st.warning(
            "Experimental mode selected. Treat smoke/fire outputs as candidate detections and validate them manually."
        )
    else:
        st.success("Stable mode selected. Prioritize vegetation and broad surface classes in the demo narrative.")
