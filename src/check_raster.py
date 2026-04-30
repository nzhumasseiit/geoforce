import rasterio
from pathlib import Path
import sys

def check_geotiff(file_path):
    p = Path(file_path)
    
    if not p.exists():
        print(f"Ошибка: Файл не найден по пути {file_path}")
        return

    try:
        with rasterio.open(file_path) as src:
            print("\n" + "=" * 40)
            print(f"📊 METADATA FOR: {p.name}")
            print("=" * 40)
            
            print(f"Dimensions:  {src.width} x {src.height}")
            print(f"Bands:       {src.count}")
            print(f"CRS:         {src.crs if src.crs else '⚠️ NOT DEFINED'}")
            
            #bounds
            b = src.bounds
            print(f"Bounds:      Left:{b.left:.5f}, Bottom:{b.bottom:.5f}, Right:{b.right:.5f}, Top:{b.top:.5f}")
            
            print(f"\nTransform (Affine):\n{src.transform}")

            print("\nColor Interpretation:")
            for i, ci in enumerate(src.colorinterp, 1):
                #band 4 is mask
                tag = "<< VALID MASK" if i == 4 else ""
                print(f"  Band {i}: {ci.name} {tag}")
            
            print("=" * 40)

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        #taking path from arg
        check_geotiff(sys.argv[1])
    else:
        #default path if noting was entered
        default_path = "/Users/nurayzhumasseiit/hardware challenge/geoai-hackathon/data/raw/almaty/Almaty_1.tif"
        print(f"No file provided. Checking default: {default_path}")
        check_geotiff(default_path)
