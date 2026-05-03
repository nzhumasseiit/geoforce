import streamlit as st
import subprocess
import shutil
import zipfile
from pathlib import Path

st.set_page_config(page_title="GeoForce Roof Detection", layout="wide")
st.title("GeoForce: Building Roof Detection from Satellite TIFFs")


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
            shutil.rmtree(path)


uploaded = st.file_uploader("Upload ZIP containing GeoTIFF", type=["zip"])

if uploaded:
    raw_dir = Path("data/raw/streamlit")
    tiles_dir = Path("data/tiles/streamlit")
    masks_dir = Path("outputs/masks/streamlit")
    masks_obia_dir = Path("outputs/masks_obia/streamlit")
    out_geo_dir = Path("outputs/geojson/streamlit")

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
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

    tif_path = tif_files[0]
    st.success(f"Found GeoTIFF: {tif_path}")

    if st.button("Run pipeline"):
        reset_pipeline_dirs([tiles_dir, masks_dir, masks_obia_dir, out_geo_dir])

        with st.spinner("Running tiling..."):
            run_cmd(["python", "src/tiling.py", str(tif_path), "--output", str(tiles_dir)])

        masks_dir.mkdir(parents=True, exist_ok=True)
        with st.spinner("Running rule masks..."):
            run_cmd([
                "python",
                "src/rules.py",
                str(tiles_dir),
                "--output",
                str(masks_dir),
            ])

        st.write("Rule mask index exists:", (masks_dir / "mask_index.json").exists())
        st.write("OBIA index exists:", (masks_obia_dir / "mask_index.json").exists())

        with st.spinner("Running OBIA..."):
            run_cmd([
                "python",
                "src/obia.py",
                str(masks_dir / "mask_index.json"),
                "--output",
                str(masks_obia_dir),
            ])

        out_geo_dir.mkdir(parents=True, exist_ok=True)
        objects_path = out_geo_dir / "objects.geojson"
        objects_clean_path = out_geo_dir / "objects_clean.geojson"
        summary_csv = out_geo_dir / "summary_metrics.csv"

        with st.spinner("Polygonizing results..."):
            run_cmd(
                [
                    "python",
                    "src/polygonize.py",
                    str(masks_obia_dir / "mask_index.json"),
                    "--output",
                    str(objects_path),
                ]
            )

        with st.spinner("Post-processing & metrics..."):
            run_cmd(
                [
                    "python",
                    "src/postprocess.py",
                    str(objects_path),
                    "--output",
                    str(objects_clean_path),
                    "--metrics",
                    str(summary_csv),
                ]
            )

        st.write("GeoJSON outputs:", list(out_geo_dir.glob("*.geojson")))

        st.success("Pipeline finished!")

        geojson_files = list(out_geo_dir.glob("objects_clean.geojson"))

        if geojson_files:
            geojson_path = max(geojson_files, key=lambda p: p.stat().st_mtime)

            st.download_button(
                "Download GeoJSON",
                geojson_path.read_bytes(),
                file_name="detected_buildings.geojson",
                mime="application/geo+json"
            )
        else:
            st.warning("GeoJSON not found yet.")
