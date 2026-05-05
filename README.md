# GeoForce

Прототип для кейса NURIS: разбиение GeoTIFF на тайлы, выделение классов местности (растительность, кровли, асфальт и т.д.) и экспорт слоя GeoJSON (+ сводные показатели).

Ссылка на исследовательский блокнот Kaggle (по желанию):  


## Dependencies

- **Python** 3.10–3.12 (разумная цель воспроизводимости; на 3.11 проверено вручную).
- GDAL/rasterio: на macOS при ошибках сборки см. официальные инструкции `rasterio` / GDAL wheel.

Установка в виртуальное окружение из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Пайплайн (CLI)

Подставьте пути к вашему GeoTIFF и папке тайлов.

```bash
# 1) Тайлы (PNG + метаданные с CRS и transform)
python src/tiling.py path/to/image.tif --output data/tiles/my_run

# 2) Эвристические маски по тайлам
python src/rules.py data/tiles/my_run --output outputs/masks/my_run

# 3) Объединение по суперпикселям (OBIA)
python src/obia.py outputs/masks/my_run/mask_index.json --output outputs/masks_obia/my_run

# 4) Полигоны в координатах сцены → GeoJSON (WGS 84 для выдачи в QGIS и т.п.)
python src/polygonize.py outputs/masks_obia/my_run/mask_index.json \
  --output outputs/geojson/my_run/objects.geojson

# 5) Постобработка метрик формы и сводная таблица по классам
python src/postprocess.py outputs/geojson/my_run/objects.geojson \
  --output outputs/geojson/my_run/objects_clean.geojson \
  --metrics outputs/geojson/my_run/summary_metrics.csv
```

Опционально: проверить метаданные растрового файла — `python src/check_raster.py path/to/file.tif`.  
Опционально: детекция YOLO — `python src/yolo_infer.py data/tiles/my_run --weights models/best.pt`.

## Приложение Streamlit

Запуск из корня проекта:

```bash
streamlit run streamlit_app.py
```

Загрузите ZIP с одним или несколькими GeoTIFF внутри. При нажатии **Run pipeline** выполняются шаги 1–5 в каталоги `data/tiles/streamlit`, `outputs/.../streamlit`; готовый слой скачивается как `detected_buildings.geojson`.

## Структура выходного GeoJSON

У записей обычно есть поля: `id`, `class`, `confidence`, `source`, `tile_name`, `area_m2`, геометрия (полигоны). Слой сохраняется в географических координатах (CRS84 / WGS 84), пригоден для открытия в QGIS на подложке.

## Docker
Альтернативный способ запуска без настройки локального окружения:
```bash
docker build -t geoforce .
docker run -p 8501:8501 geoforce
```
После запуска приложение доступно по адресу `http://localhost:8501`.
