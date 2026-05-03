# GeoForce

Прототип для кейса NURIS: AOI-aware обработка GeoTIFF, emergency-oriented классификация поверхности и экспорт результата в GeoJSON / GeoPackage.

Ссылка на исследовательский блокнот Colab (по желанию):  
https://colab.research.google.com/drive/1Qp1sh2V1O69pzWe-jAC2iAJTInZCUFpu?usp=sharing

## Зависимости

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

Подставьте пути к вашему GeoTIFF, папке тайлов и при необходимости AOI-контуру.

```bash
# 1) Тайлы (PNG + метаданные с CRS и transform)
python src/tiling.py path/to/image.tif --output data/tiles/my_run --aoi path/to/aoi.geojson

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

# 6) Экспорт в GeoPackage (рекомендуется для выдачи)
python src/export.py outputs/geojson/my_run/objects_clean.geojson \
  --output outputs/exports/my_run/objects_clean.gpkg
```

Опционально: проверить метаданные растрового файла — `python src/check_raster.py path/to/file.tif`.  
YOLO-ветка оставлена как архитектурное расширение для будущего дообучения на ЧС-данных, но основной оцениваемый пайплайн — `rule-based + OBIA`.

## Приложение Streamlit

Запуск из корня проекта:

```bash
streamlit run streamlit_app.py
```

Можно загрузить ZIP с одним или несколькими GeoTIFF внутри или указать локальный путь к `.tif/.tiff`. Дополнительно можно задать путь к AOI-вектору (`.geojson/.gpkg/.shp`). При нажатии **Run pipeline** выполняются шаги 1–6 в каталоги `data/tiles/streamlit`, `outputs/.../streamlit`; готовые слои доступны для скачивания как GeoJSON и GeoPackage.

## Классы

- `vegetation`
- `impervious_surface`
- `smoke_plume`
- `active_fire`
- `water`
- `bare_soil`

`shadow_ignore` используется только как внутренний QA-класс и не предназначен для финальной выдачи.

## Структура выходного GeoJSON

У записей обычно есть поля: `id`, `class`, `confidence`, `confidence_method`, `source`, `tile_name`, `area_m2`, `valid_ratio`, геометрия (полигоны). Слой сохраняется в географических координатах (CRS84 / WGS 84), пригоден для открытия в QGIS на подложке.

Важно: `confidence` в текущей версии — не вероятность модели, а геометрический прокси уверенности (`confidence_method = geometry_proxy`) на основе площади объекта и доли валидных пикселей.
