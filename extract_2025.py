import pandas as pd
import numpy as np
from datetime import timedelta
import ee

# 1. เชื่อมต่อระบบดาวเทียม Google Earth Engine
try:
    ee.Initialize(project='gee-training-498303')
    print("🛰️ เชื่อมต่อระบบ Google Earth Engine สำเร็จ!")
except Exception as e:
    print("🔒 กำลังขอเข้าถึงสิทธิ์ผ่าน Terminal...")
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='gee-training-498303')

# 2. 🚨 อ่านไฟล์ข้อมูลภาคสนาม PCD ของปี 2025
# (จี๊ดอย่าลืมตรวจเช็กชื่อไฟล์ตาราง PCD ปี 2025 ของตัวเองในโฟลเดอร์น้าว่าสะกดแบบนี้ไหม)
INPUT_FILE_2025 = "PCD_WQ_Ubon_2025_Table.csv" 

try:
    df_raw = pd.read_csv(INPUT_FILE_2025)
    print(f"📊 โหลดไฟล์ PCD ปี 2025 สำเร็จ! พบข้อมูลทั้งหมด {len(df_raw)} แถว")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์: {INPUT_FILE_2025} ในโฟลเดอร์ D:\\Chula\\ADPC")
    print("👉 จี๊ดอย่าลืมเอาไฟล์ตารางภาคสนามของปี 2025 มาวางคู่กับสคริปต์นี้ก่อนน้า")
    exit()

def maskS2(image):
    qa = image.select("QA60")
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    no_cloud = qa.bitwiseAnd(cloudBitMask).eq(0)
    no_cirrus = qa.bitwiseAnd(cirrusBitMask).eq(0)
    mask = no_cloud.And(no_cirrus)
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"])

def addIndices(img):
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndti = img.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndssi = img.normalizedDifference(["B4", "B8"]).rename("NDSSI")
    return img.addBands([ndwi, mndwi, ndti, ndssi])

# ⭐️ ปรับช่วงเวลาค้นหาภาพดาวเทียมเป็นของปี 2025 ทั้งปี
S2_2025 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterDate("2025-01-01", "2025-12-31")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 100))
      .map(maskS2)
      .map(addIndices))

result_data = []
print("\n🚀 เริ่มสกัดข้อมูลดาวเทียมของปี 2025...")

for index, row in df_raw.iterrows():
    try:
        lon = float(row["station_lo"])
        lat = float(row["station_la"])
        point = ee.Geometry.Point([lon, lat])

        date = pd.to_datetime(row["Date"])
        start = (date - timedelta(days=15)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=15)).strftime("%Y-%m-%d")

        filtered = S2_2025.filterBounds(point).filterDate(start, end).sort("CLOUDY_PIXEL_PERCENTAGE")
        if filtered.size().getInfo() == 0:
            print(f"  -> แถว {index+1}: ❌ ไม่พบรอบวงโคจรช่วงนี้")
            continue

        img = filtered.first()
        value = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10, maxPixels=1e13).getInfo()

        if value is None or value.get('B2') is None:
            print(f"  -> แถว {index+1}: ☁️ ข้ามพิกเซลที่ติดเมฆทึบ")
            continue

        row_data = row.to_dict()
        row_data.update(value)
        result_data.append(row_data)
        print(f"  -> แถว {index+1}: ✅ สกัดข้อมูลดาวเทียมปี 2025 สำเร็จ!")
    except Exception as e:
        print(f"  -> แถว {index+1}: ⚠️ เกิดปัญหา: {e}")
        continue

df_extracted = pd.DataFrame(result_data)
if len(df_extracted) > 0:
    # ⭐️ เซฟเป็นไฟล์แยกสำหรับปี 2025 โดยเฉพาะ
    OUTPUT_FILE_2025 = "Sentinel2_Extract_2025.csv"
    df_extracted.to_csv(OUTPUT_FILE_2025, index=False)
    print(f"\n🎉 สำเร็จ! ได้ไฟล์สกัดปี 2025 แล้วที่: '{OUTPUT_FILE_2025}'")