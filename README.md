Pipeline Overview
TIFF → Tiling → Weak Masks → OBIA → Polygonization → Postprocessing
Steps:
Tiling: Splits large satellite .tif into manageable tiles
Rule-based Masking (rules.py): Generates weak segmentation masks using color + heuristics
OBIA (obia.py): Superpixel segmentation (SLIC), assigns class per region
Polygonization: Converts masks → vector polygons (GeoJSON)
Postprocessing: Cleans noise and improves class consistency

📂 Project Structure
geoai-hackathon/
│
├── data/
│   ├── raw/              # original TIFFs
│   ├── tiles/            # generated tiles
│
├── outputs/
│   ├── masks/            # segmentation masks
│   ├── geojson/          # final polygons
│
├── src/
│   ├── tiling.py
│   ├── rules.py
│   ├── obia.py
│   ├── polygonize.py
│
├── streamlit_app.py      # demo UI
├── requirements.txt
└── README.md

⚙️ Installation
git clone <your-repo>
cd geoai-hackathon

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
▶️ Usage
1. Tile satellite image
python src/tiling.py data/raw/almaty/Almaty_1.tif
2. Generate masks
python src/rules.py data/tiles --output outputs/masks
3. Run OBIA
python src/obia.py outputs/masks/mask_index.json
4. Polygonize
python src/polygonize.py outputs/obia_output.geojson

🗺️ Output
Final result: GeoJSON polygons
Classes:
rooftop
vegetation
paved_area
shadow_ignore
Visualize in:
QGIS
GeoJSON viewers
Streamlit app

Why Hybrid Approach?
Pure deep learning:
needs large labeled datasets
Pure rule-based:
lacks generalization
This system combines both:
fast classical segmentation
optional ML refinement (YOLO-ready)
Future Work
YOLO rooftop detection integration
Multi-city scaling
Real-time web deployment
Dockerization
🏆 Use Cases
Urban planning
Infrastructure monitoring
Disaster response
Smart city analytics
Link to Colab notebook: https://colab.research.google.com/drive/1Qp1sh2V1O69pzWe-jAC2iAJTInZCUFpu?usp=sharing
