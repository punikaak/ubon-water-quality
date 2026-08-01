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

# 🚨 ปรับตามชื่อไฟล์ตารางข้อมูลของจี๊ดเรียบร้อยแล้วครับ
INPUT_FILE_SAKON = "PCD_WQ_Mekong_2024_Table.csv" 

try:
    df_raw = pd.read_csv(INPUT_FILE_SAKON)
    print(f"📊 โหลดไฟล์ข้อมูลสกลนคร/แม่โขงสำเร็จ! พบข้อมูลทั้งหมด {len(df_raw)} แถว")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์: {INPUT_FILE_SAKON} ในโฟลเดอร์รันสคริปต์")
    print("👉 จี๊ดอย่าลืมเช็กว่าเอาไฟล์นี้ไปวางในโฟลเดอร์ D:\\Chula\\ADPC คู่กับสคริปต์แล้วหรือยังน้า")
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

# ค้นหาภาพดาวเทียมครอบคลุมพิกัดและช่วงเวลาข้อมูล
S2_Collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 100))
      .map(maskS2)
      .map(addIndices))

result_data = []
print("\n🚀 เริ่มสกัดข้อมูลดาวเทียม Sentinel-2 แถบสกลนคร/แม่โขง...")

for index, row in df_raw.iterrows():
    try:
        lon = float(row["station_lo"])
        lat = float(row["station_la"])
        point = ee.Geometry.Point([lon, lat])

        date = pd.to_datetime(row["Date"])
        start = (date - timedelta(days=15)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=15)).strftime("%Y-%m-%d")

        filtered = S2_Collection.filterBounds(point).filterDate(start, end).sort("CLOUDY_PIXEL_PERCENTAGE")
        if filtered.size().getInfo() == 0:
            print(f"  -> แถว {index+1}: ❌ ไม่พบภาพดาวเทียมช่วงวันที่นี้")
            continue

        img = filtered.first()
        value = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10, maxPixels=1e13).getInfo()

        if value is None or value.get('B2') is None:
            print(f"  -> แถว {index+1}: ☁️ ติดเมฆหนา ข้ามไปก่อน")
            continue

        row_data = row.to_dict()
        row_data.update(value)
        result_data.append(row_data)
        print(f"  -> แถว {index+1}: ✅ สกัดข้อมูลแบนด์และดัชนีสำเร็จ!")
    except Exception as e:
        print(f"  -> แถว {index+1}: ⚠️ เกิดข้อผิดพลาดชั่วคราว: {e}")
        continue

df_extracted = pd.DataFrame(result_data)
if len(df_extracted) > 0:
    # เซฟแยกเป็นไฟล์ผลลัพธ์ของฝั่งสกลนคร/แม่โขง
    OUTPUT_FILE_SAKON = "Sentinel2_Extract_Sakon.csv"
    df_extracted.to_csv(OUTPUT_FILE_SAKON, index=False)
    print(f"\n🎉 เรียบร้อย! ดึงค่าดาวเทียมเสร็จแล้ว ได้ไฟล์ใหม่ชื่อ: '{OUTPUT_FILE_SAKON}'")