import streamlit as st
import subprocess
from pathlib import Path
import zipfile
from pathlib import Path
import shutil

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

uploaded = st.file_uploader("Upload ZIP containing GeoTIFF", type=["zip"])

if uploaded:
    raw_dir = Path("data/raw/streamlit")

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
        with st.spinner("Running tiling..."):
            run_cmd(["python", "src/tiling.py", str(tif_path), "--output", "data/tiles/streamlit"])
        

        Path("outputs/masks/streamlit").mkdir(parents=True, exist_ok=True)
        with st.spinner("Running rule masks..."):
            run_cmd([
                "python",
                "src/rules.py",
                "data/tiles/streamlit",
                "--output",
                "outputs/masks/streamlit"
            ])

        st.write("Rule mask index exists:", Path("outputs/masks/streamlit/mask_index.json").exists())
        st.write("OBIA index exists:", Path("outputs/masks_obia/streamlit/mask_index.json").exists())

        with st.spinner("Running OBIA..."):
            run_cmd(["python", "src/obia.py", "outputs/masks/streamlit/mask_index.json", "--output", "outputs/masks_obia/streamlit"])

        with st.spinner("Polygonizing results..."):
            run_cmd(["python", "src/polygonize.py", "outputs/masks_obia/streamlit/mask_index.json", "--output", "outputs/geojson/streamlit"])

        st.write("GeoJSON search all outputs:")
        st.write(list(Path("outputs").rglob("*.geojson")))

        st.success("Pipeline finished!")

        geojson_files = list(Path("outputs/geojson").rglob("*objects_clean.geojson"))

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