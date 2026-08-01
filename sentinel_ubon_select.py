import os
import zipfile
import io
import requests
import numpy as np
import rasterio
import joblib
import ee

# =================================================================
# 🛰️ 1. โหลดภาพ Sentinel-2 เดือน 11/2024 (แบบกันตายทุกสถานการณ์)
# =================================================================
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

roi = ee.Geometry.Rectangle([104.6, 15.1, 105.6, 15.4])
start_date = '2024-11-01'
end_date = '2024-11-30'

print("⏳ กำลังค้นหาและขอลิงก์ดาวน์โหลดจาก Google Earth Engine...")

s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .sort('CLOUDY_PIXEL_PERCENTAGE'))

image_selected = s2_col.first().select(['B2', 'B3', 'B4', 'B8', 'B11'])

# 🛠️ ลดความละเอียดเหลือ 40m เพื่อให้ขนาดไฟล์ผ่านเกณฑ์ GEE
download_url = image_selected.getDownloadURL({
    'scale': 40,
    'crs': 'EPSG:4326',
    'region': roi,
    'filePerBand': False,
    'format': 'GEO_TIFF'
})

print("📥 กำลังดาวน์โหลดไฟล์ภาพดิบเข้าเครื่อง...")
response = requests.get(download_url)
input_tiff = "sentinel2_2024_11.tif"

if response.status_code == 200:
    try:
        # ลองแตกไฟล์ดูว่ามาเป็น Zip หรือเปล่า
        z = zipfile.ZipFile(io.BytesIO(response.content))
        tif_name = [name for name in z.namelist() if name.endswith('.tif')][0]
        z.extract(tif_name, path=".")
        if os.path.exists(input_tiff):
            os.remove(input_tiff)
        os.rename(tif_name, input_tiff) # เปลี่ยนชื่อไฟล์
    except zipfile.BadZipFile:
        # ถ้าไม่ใช่ Zip แสดงว่าส่งเป็น Tiff มาตรงๆ ให้เซฟลงเครื่องเลย
        with open(input_tiff, 'wb') as f:
            f.write(response.content)
            
    print(f"✅ บันทึกไฟล์ภาพดาวเทียมดิบสำเร็จ: '{input_tiff}'\n")
else:
    print(f"❌ ดาวน์โหลดไม่สำเร็จ: {response.text}")
    exit() # หยุดการทำงานถ้าโหลดภาพไม่ได้

# =================================================================
# 🧠 2. โหลดสมองโมเดล MLP และแปลงภาพเป็นแผนที่ Turbidity NTU
# =================================================================
mlp = joblib.load("best_model_neural_network_mlp.pkl")
scaler = joblib.load("scaler_ubon_final.pkl") # 🛠️ ถ้าเครื่องจี๊ดชื่ออื่น อย่าลืมแก้ตรงนี้น้า!
output_tiff = "turbidity_map_mlp_2024_11.tif"

def calculate_index(band1, band2):
    with np.errstate(divide='ignore', invalid='ignore'):
        index = (band1 - band2) / (band1 + band2)
        index = np.nan_to_num(index, nan=0.0, posinf=0.0, neginf=0.0)
    return index

print("🔮 กำลังประมวลผลพิกเซลภาพด้วยโมเดล MLP Regressor...")
with rasterio.open(input_tiff) as src:
    profile = src.profile
    
    b2 = src.read(1) / 10000.0
    b3 = src.read(2) / 10000.0
    b4 = src.read(3) / 10000.0
    b8 = src.read(4) / 10000.0
    b11 = src.read(5) / 10000.0
    
    height, width = b2.shape

    ndwi = calculate_index(b3, b8)
    mndwi = calculate_index(b3, b11)
    ndti = calculate_index(b4, b3)
    ndssi = calculate_index(b8, b4)

    X_pixels = np.column_stack([
        b2.flatten(), b3.flatten(), b4.flatten(), b8.flatten(), 
        ndwi.flatten(), mndwi.flatten(), ndti.flatten(), ndssi.flatten()
    ])

    X_pixels_scaled = scaler.transform(X_pixels)

    predicted_flat = mlp.predict(X_pixels_scaled)
    predicted_flat = np.clip(predicted_flat, a_min=0.0, a_max=None) 

    turbidity_map = predicted_flat.reshape((height, width))

profile.update(
    dtype=rasterio.float32,
    count=1,
    nodata=0.0
)

with rasterio.open(output_tiff, "w", **profile) as dst:
    dst.write(turbidity_map.astype(rasterio.float32), 1)

print("=================================================================")
print(f"🏆 ประมวลผลเสร็จสมบูรณ์! แผนที่ความขุ่นน้ำ NTU ถูกบันทึกไว้ที่:")
print(f"📂 ไฟล์ภาพ: '{output_tiff}'")
print("=================================================================")