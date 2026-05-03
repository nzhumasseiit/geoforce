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
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def prepare_input_tif() -> Path | None:
    source_mode = st.radio(
        "Input source",
        ("ZIP upload", "Local path"),
        horizontal=True,
    )

    if source_mode == "ZIP upload":
        uploaded = st.file_uploader("Upload ZIP containing GeoTIFF", type=["zip"])
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

        tif_path = tif_files[0]
        st.success(f"Found GeoTIFF: {tif_path}")
        return tif_path

    local_path = st.text_input(
        "Local GeoTIFF path",
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
        tif_path = tif_files[0]
    elif tif_path.suffix.lower() not in {".tif", ".tiff"}:
        st.error("Please provide a .tif/.tiff file or a folder containing GeoTIFFs.")
        st.stop()

    st.success(f"Using local GeoTIFF: {tif_path}")
    return tif_path


tif_path = prepare_input_tif()

if tif_path is not None:
    raw_dir = Path("data/raw/streamlit")
    tiles_dir = Path("data/tiles/streamlit")
    masks_dir = Path("outputs/masks/streamlit")
    masks_obia_dir = Path("outputs/masks_obia/streamlit")
    out_geo_dir = Path("outputs/geojson/streamlit")

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
