import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


APP_TITLE = "GeoForce: Satellite Imagery to GIS Layers"
APP_SUBTITLE = "AOI-aware satellite image processing with QGIS-ready GeoJSON and GeoPackage outputs."
PIPELINE_VISUAL = "GeoTIFF -> Tiling -> CV Masks -> OBIA -> Polygons -> GIS Export"

STABLE_CLASSES = {"vegetation", "impervious_surface", "bare_soil"}
CLASS_LABELS = {
    "vegetation": "Vegetation",
    "impervious_surface": "Built-up / Impervious Surface",
    "bare_soil": "Bare Soil",
}
YOLO_WEIGHTS = Path("models/emergency.pt")
YOLO_OUTPUT_ROOT = Path("outputs/yolo/emergency")
PREVIEW_COLORS = {
    "vegetation": np.array([0, 210, 140], dtype=np.uint8),
    "impervious_surface": np.array([0, 194, 255], dtype=np.uint8),
    "bare_soil": np.array([157, 107, 255], dtype=np.uint8),
}


st.set_page_config(page_title=APP_TITLE, layout="wide")


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .app-subtitle {
            color: #9fb0c0;
            margin-top: -0.35rem;
            margin-bottom: 1.25rem;
            font-size: 1rem;
        }
        .panel-card {
            background: rgba(15, 22, 42, 0.94);
            border: 1px solid rgba(111, 103, 255, 0.18);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .kpi-card {
            background: linear-gradient(180deg, rgba(18, 27, 51, 0.98) 0%, rgba(8, 13, 28, 0.98) 100%);
            border: 1px solid rgba(111, 103, 255, 0.16);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            min-height: 118px;
            box-shadow: 0 14px 36px rgba(6, 10, 24, 0.32);
        }
        .kpi-label {
            color: #94a7b8;
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            color: #E6EDF3;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
        }
        .legend-block {
            background: rgba(18, 27, 51, 0.88);
            border: 1px solid rgba(111, 103, 255, 0.14);
            border-radius: 14px;
            padding: 0.95rem 1rem;
        }
        .legend-title {
            color: #00C2FF;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .legend-list {
            margin: 0;
            padding-left: 1rem;
            color: #d6dfe7;
        }
        .pipeline-strip {
            background: linear-gradient(90deg, rgba(0, 194, 255, 0.12) 0%, rgba(18, 27, 51, 0.96) 65%, rgba(111, 103, 255, 0.16) 100%);
            border: 1px solid rgba(0, 194, 255, 0.20);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: #dbe5ee;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-bottom: 1rem;
        }
        .preview-frame {
            background: rgba(18, 27, 51, 0.88);
            border: 1px solid rgba(111, 103, 255, 0.14);
            border-radius: 16px;
            padding: 0.85rem;
            min-height: 100%;
        }
        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00C2FF 0%, #6F67FF 100%);
            color: #08111d;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_cmd(cmd, fatal=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and fatal:
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
        uploaded = st.sidebar.file_uploader("Input GeoTIFF", type=["zip"])
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


def format_area(value: float) -> str:
    return f"{value:,.0f} m²"


def pretty_class_name(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name.replace("_", " ").title())


def filter_stable_outputs(gdf: gpd.GeoDataFrame, metrics_df: pd.DataFrame):
    stable_gdf = gdf[gdf["class"].isin(STABLE_CLASSES)].copy() if "class" in gdf.columns else gdf.copy()
    if not metrics_df.empty and "class" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["class"].isin(STABLE_CLASSES)].copy()
    return stable_gdf, metrics_df


def parse_emergency_candidates(yolo_run_dir: Path) -> pd.DataFrame:
    labels_dir = yolo_run_dir / "labels"
    rows = []

    if not labels_dir.exists():
        return pd.DataFrame()

    for label_path in sorted(labels_dir.glob("*.txt")):
        with open(label_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                confidence = float(parts[5]) if len(parts) >= 6 else None
                rows.append(
                    {
                        "tile_name": label_path.stem,
                        "candidate_id": f"{label_path.stem}_{idx}",
                        "candidate_label": "Smoke / fire candidate",
                        "confidence_hint": confidence,
                    }
                )

    return pd.DataFrame(rows)


def load_binary_mask(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def build_preview_images(tile_root: Path, obia_index_path: Path, preview_dir: Path):
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not obia_index_path.exists():
        return {"source_preview": None, "overlay_preview": None, "tile_name": None}

    with open(obia_index_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    by_tile = {}
    for record in records:
        cls = record["class"]
        if cls not in STABLE_CLASSES:
            continue
        by_tile.setdefault(record["tile_name"], {})[cls] = record["mask_path"]

    best_tile = None
    best_score = -1
    best_masks = None

    for tile_name, masks in by_tile.items():
        score = 0
        for cls, mask_path in masks.items():
            score += int(load_binary_mask(mask_path).sum())
        if score > best_score:
            best_score = score
            best_tile = tile_name
            best_masks = masks

    if best_tile is None:
        return {"source_preview": None, "overlay_preview": None, "tile_name": None}

    image_path = tile_root / "images" / f"{best_tile}.png"
    if not image_path.exists():
        return {"source_preview": None, "overlay_preview": None, "tile_name": best_tile}

    base = np.array(Image.open(image_path).convert("RGB"))
    overlay = base.copy()

    for cls, color in PREVIEW_COLORS.items():
        mask_path = best_masks.get(cls)
        if not mask_path:
            continue
        mask = load_binary_mask(mask_path)
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)

    source_out = preview_dir / f"{best_tile}_source.png"
    overlay_out = preview_dir / f"{best_tile}_overlay.png"
    Image.fromarray(base).save(source_out)
    Image.fromarray(overlay).save(overlay_out)

    return {
        "source_preview": str(source_out),
        "overlay_preview": str(overlay_out),
        "tile_name": best_tile,
    }


def run_experimental_yolo(tile_root: Path):
    reset_pipeline_dirs([YOLO_OUTPUT_ROOT])
    YOLO_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not YOLO_WEIGHTS.exists():
        return {
            "enabled": False,
            "message": "Experimental smoke/fire YOLO module is optional. Add trained weights to models/emergency.pt to enable it.",
            "output_dir": str(YOLO_OUTPUT_ROOT),
            "error": None,
        }

    with st.spinner("Running optional experimental emergency YOLO..."):
        result = run_cmd(
            [
                "python",
                "src/yolo_infer.py",
                str(tile_root),
                "--weights",
                str(YOLO_WEIGHTS),
                "--output",
                str(YOLO_OUTPUT_ROOT),
                "--conf",
                "0.10",
            ],
            fatal=False,
        )

    run_dir = YOLO_OUTPUT_ROOT / "streamlit_preds"
    summary_df = parse_emergency_candidates(run_dir)

    if result.returncode != 0:
        return {
            "enabled": True,
            "message": "Experimental smoke/fire YOLO module could not complete. Stable GIS extraction is still available.",
            "output_dir": str(run_dir),
            "error": result.stderr or result.stdout,
            "summary_df": summary_df,
        }

    return {
        "enabled": True,
        "message": "Experimental smoke/fire YOLO module finished. Outputs are labeled as candidate detections only.",
        "output_dir": str(run_dir),
        "error": None,
        "summary_df": summary_df,
    }


def run_pipeline(tif_path: Path, aoi_path: str, min_area: float, detection_mode: str):
    tiles_dir = Path("data/tiles/streamlit")
    masks_dir = Path("outputs/masks/streamlit")
    masks_obia_dir = Path("outputs/masks_obia/streamlit")
    out_geo_dir = Path("outputs/geojson/streamlit")
    exports_dir = Path("outputs/exports/streamlit")
    previews_dir = Path("outputs/previews/streamlit")

    reset_pipeline_dirs([tiles_dir, masks_dir, masks_obia_dir, out_geo_dir, exports_dir, previews_dir])

    with st.spinner("Running tiling..."):
        cmd = ["python", "src/tiling.py", str(tif_path), "--output", str(tiles_dir)]
        if aoi_path:
            cmd.extend(["--aoi", aoi_path])
        run_cmd(cmd)

    masks_dir.mkdir(parents=True, exist_ok=True)
    with st.spinner("Running CV masks..."):
        run_cmd(["python", "src/rules.py", str(tiles_dir), "--output", str(masks_dir)])

    with st.spinner("Running OBIA smoothing..."):
        run_cmd(["python", "src/obia.py", str(masks_dir / "mask_index.json"), "--output", str(masks_obia_dir)])

    out_geo_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    objects_path = out_geo_dir / "objects.geojson"
    objects_clean_path = out_geo_dir / "objects_clean.geojson"
    summary_csv = out_geo_dir / "summary_metrics.csv"
    stable_geojson_path = out_geo_dir / "objects_stable.geojson"
    stable_metrics_csv = out_geo_dir / "summary_metrics_stable.csv"
    objects_gpkg = exports_dir / "objects_stable.gpkg"

    with st.spinner("Polygonizing masks..."):
        run_cmd(["python", "src/polygonize.py", str(masks_obia_dir / "mask_index.json"), "--output", str(objects_path)])

    with st.spinner("Post-processing stable GIS layers..."):
        run_cmd(
            [
                "python",
                "src/postprocess.py",
                str(objects_path),
                "--output",
                str(objects_clean_path),
                "--metrics",
                str(summary_csv),
                "--min-area",
                str(min_area),
            ]
        )

    stable_gdf = gpd.read_file(objects_clean_path)
    metrics_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    stable_gdf, stable_metrics_df = filter_stable_outputs(stable_gdf, metrics_df)

    stable_gdf.to_file(stable_geojson_path, driver="GeoJSON")
    stable_metrics_df.to_csv(stable_metrics_csv, index=False)

    preview_info = build_preview_images(tiles_dir, masks_obia_dir / "mask_index.json", previews_dir)

    with st.spinner("Exporting GeoPackage..."):
        run_cmd(["python", "src/export.py", str(stable_geojson_path), "--output", str(objects_gpkg)])

    emergency = {
        "enabled": False,
        "message": "Experimental smoke/fire YOLO module is optional. Add trained weights to models/emergency.pt to enable it.",
        "output_dir": str(YOLO_OUTPUT_ROOT),
        "error": None,
        "summary_df": pd.DataFrame(),
    }
    if detection_mode == "Experimental emergency AI":
        emergency = run_experimental_yolo(tiles_dir)

    return {
        "tif_path": str(tif_path),
        "aoi_path": aoi_path or None,
        "detection_mode": detection_mode,
        "stable_geojson_path": str(stable_geojson_path),
        "stable_metrics_csv": str(stable_metrics_csv),
        "objects_gpkg": str(objects_gpkg),
        "preview_source": preview_info["source_preview"],
        "preview_overlay": preview_info["overlay_preview"],
        "preview_tile_name": preview_info["tile_name"],
        "emergency": emergency,
    }


def load_results():
    last_run = st.session_state.get("last_run")
    if not last_run:
        return None

    stable_geojson = Path(last_run["stable_geojson_path"])
    stable_metrics_csv = Path(last_run["stable_metrics_csv"])
    if not stable_geojson.exists():
        return None

    gdf = gpd.read_file(stable_geojson)
    metrics_df = pd.read_csv(stable_metrics_csv) if stable_metrics_csv.exists() else pd.DataFrame()

    if "class" in gdf.columns:
        gdf["class_label"] = gdf["class"].map(pretty_class_name)
    if not metrics_df.empty and "class" in metrics_df.columns:
        metrics_df["class_label"] = metrics_df["class"].map(pretty_class_name)

    return {
        "meta": last_run,
        "gdf": gdf,
        "metrics_df": metrics_df,
        "emergency": last_run.get("emergency", {}),
    }


def render_kpis(gdf: gpd.GeoDataFrame):
    objects_detected = len(gdf)
    classes_found = int(gdf["class"].nunique()) if "class" in gdf.columns else 0
    total_area = float(gdf["area_m2"].sum()) if "area_m2" in gdf.columns else 0.0
    export_formats = 2

    cards = [
        ("Objects detected", f"{objects_detected}"),
        ("Classes found", f"{classes_found}"),
        ("Total mapped area", format_area(total_area)),
        ("Export formats", f"{export_formats}"),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_legend():
    st.markdown(
        """
        <div class="legend-grid">
            <div class="legend-block">
                <div class="legend-title">Stable</div>
                <ul class="legend-list">
                    <li>Vegetation</li>
                    <li>Built-up / Impervious Surface</li>
                    <li>Bare Soil</li>
                </ul>
            </div>
            <div class="legend-block">
                <div class="legend-title">Experimental</div>
                <ul class="legend-list">
                    <li>Smoke-like candidate</li>
                    <li>Fire-like candidate</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_visual():
    st.markdown(f'<div class="pipeline-strip">{PIPELINE_VISUAL}</div>', unsafe_allow_html=True)


def get_input_preview_path(path_str: str) -> Path | None:
    path = Path(path_str)
    if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return path
    return None


inject_css()
st.title(APP_TITLE)
st.markdown(f'<div class="app-subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis Settings")
    source_mode = st.radio("Input mode", ("ZIP upload", "Local path"))
    tif_path = prepare_input_tif(source_mode)
    aoi_path = st.text_input("Optional AOI path", placeholder="/absolute/path/to/aoi.geojson").strip()
    detection_mode = st.selectbox("Detection mode", ["Stable GIS extraction", "Experimental emergency AI"])
    min_area = st.number_input("Min object area", min_value=1.0, value=25.0, step=25.0)
    run_clicked = st.button("Run Analysis", type="primary", disabled=tif_path is None)

if tif_path is not None:
    st.success(f"Input ready: {tif_path}")
else:
    st.info("Choose a GeoTIFF source in the sidebar to begin.")

render_pipeline_visual()
render_legend()

st.markdown(
    """
    <div class="panel-card">
        <strong>Stable mode exports only validated land-cover classes.</strong>
        Experimental emergency AI is separated and should be treated as candidate detections.
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Confidence is a geometry-based proxy score, not a neural network probability.")

if run_clicked and tif_path is not None:
    st.session_state["last_run"] = run_pipeline(tif_path, aoi_path, min_area, detection_mode)
    st.success("Analysis finished.")

results = load_results()

tabs = st.tabs(["Summary", "Detected Objects", "Analytics", "Export", "Experimental AI"])

with tabs[0]:
    st.subheader("Summary")
    if results is None:
        st.write("Run the analysis to see stable GIS layers, summary metrics, and export options.")
    else:
        st.markdown("### Summary")
        render_kpis(results["gdf"])
        st.markdown("---")
        st.markdown(
            """
            <div class="panel-card">
                <strong>Stable demo workflow:</strong> GeoTIFF is tiled, processed with CV masks, smoothed by OBIA,
                converted to polygons, and exported as QGIS-ready GeoJSON and GeoPackage.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            **Input file:** `{results["meta"]["tif_path"]}`  
            **Mode:** {results["meta"]["detection_mode"]}  
            **AOI:** {results["meta"]["aoi_path"] or "None"}  
            **Confidence:** `geometry_proxy`
            """
        )

with tabs[1]:
    st.subheader("Detected Objects")
    if results is None:
        st.write("No results yet.")
    else:
        gdf = results["gdf"].copy()
        st.markdown("### Visual preview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            preview_path = Path(results["meta"]["preview_source"]) if results["meta"].get("preview_source") else None
            if preview_path is not None:
                st.markdown('<div class="preview-frame">', unsafe_allow_html=True)
                st.image(str(preview_path), use_container_width=True)
                st.caption(f"Representative tile: {results['meta'].get('preview_tile_name')}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Input preview is not available directly in Streamlit for this file, but the source path is preserved for QGIS and local inspection.")
        with col2:
            st.subheader("Detected Objects (Preview)")
            overlay_path = Path(results["meta"]["preview_overlay"]) if results["meta"].get("preview_overlay") else None
            if overlay_path is not None:
                st.markdown('<div class="preview-frame">', unsafe_allow_html=True)
                st.image(str(overlay_path), use_container_width=True)
                st.caption("Stable class overlay preview")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <div class="panel-card">
                        Stable output is prepared for QGIS overlay and export.
                        Use this panel together with the class distribution below to explain what the pipeline extracted.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### Class distribution")
        class_counts = gdf["class_label"].value_counts()
        st.bar_chart(class_counts)

        st.markdown("---")
        st.write("Stable GIS objects preview")
        preview_cols = [c for c in ["id", "class_label", "confidence", "confidence_method", "area_m2", "tile_name"] if c in gdf.columns]
        st.dataframe(gdf[preview_cols].head(50), use_container_width=True)
        st.markdown("---")
        st.write("Stable classes detected")
        st.dataframe(class_counts.rename_axis("class").reset_index(name="objects"), use_container_width=True)

with tabs[2]:
    st.subheader("Analytics")
    if results is None:
        st.write("No metrics yet.")
    else:
        metrics_df = results["metrics_df"]
        if metrics_df.empty:
            st.write("Metrics table not found.")
        else:
            st.markdown("### Objects by class")
            st.bar_chart(metrics_df.set_index("class_label")["objects"])
            st.markdown("---")
            st.markdown("### Metrics table")
            metric_cols = [c for c in ["class_label", "objects", "total_area_m2", "mean_area_m2", "median_area_m2", "area_percent"] if c in metrics_df.columns]
            st.dataframe(metrics_df[metric_cols], use_container_width=True)
        st.caption("Manual QA is visual comparison in QGIS/Streamlit because no ground-truth labels are available.")

with tabs[3]:
    st.subheader("Export")
    if results is None:
        st.write("No downloadable outputs yet.")
    else:
        stable_geojson_path = Path(results["meta"]["stable_geojson_path"])
        stable_metrics_csv = Path(results["meta"]["stable_metrics_csv"])
        gpkg_path = Path(results["meta"]["objects_gpkg"])

        st.write("Stable mode downloads include only validated land-cover classes.")
        if stable_geojson_path.exists():
            st.download_button(
                "Download Stable GeoJSON",
                stable_geojson_path.read_bytes(),
                file_name="geoforce_stable.geojson",
                mime="application/geo+json",
            )
        if gpkg_path.exists():
            st.download_button(
                "Download Stable GeoPackage",
                gpkg_path.read_bytes(),
                file_name="geoforce_stable.gpkg",
                mime="application/geopackage+sqlite3",
            )
        if stable_metrics_csv.exists():
            st.download_button(
                "Download Stable Metrics CSV",
                stable_metrics_csv.read_bytes(),
                file_name="geoforce_stable_metrics.csv",
                mime="text/csv",
            )

with tabs[4]:
    st.subheader("Experimental AI")
    emergency = results["emergency"] if results is not None else {}
    st.write("Experimental smoke/fire YOLO module is optional. Add trained weights to models/emergency.pt to enable it.")
    st.write("Any experimental detections are labeled as candidate detections and are not merged into the stable GIS export.")

    if not emergency:
        st.info("Run the analysis to initialize the optional experimental module.")
    else:
        message = emergency.get("message")
        if message:
            if emergency.get("error"):
                st.warning(message)
            else:
                st.info(message)

        st.caption(f"Output directory: {emergency.get('output_dir', str(YOLO_OUTPUT_ROOT))}")

        if emergency.get("error"):
            with st.expander("Experimental module error details"):
                st.code(emergency["error"])

        summary_df = emergency.get("summary_df", pd.DataFrame())
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            candidate_count = len(summary_df)
            tile_count = summary_df["tile_name"].nunique()
            mean_conf = summary_df["confidence_hint"].dropna().mean()

            cols = st.columns(3)
            cols[0].metric("Candidate detections", str(candidate_count))
            cols[1].metric("Tiles with candidates", str(tile_count))
            cols[2].metric("Mean confidence hint", f"{mean_conf:.2f}" if pd.notna(mean_conf) else "n/a")

            st.dataframe(summary_df.head(100), use_container_width=True)
        elif emergency.get("enabled"):
            st.write("No experimental candidate detections were produced for this run.")
