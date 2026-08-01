import pandas as pd
import numpy as np
from datetime import timedelta
import re
import ee

# 1. เชื่อมต่อระบบดาวเทียม Google Earth Engine
try:
    ee.Initialize(project='gee-training-498303')
    print("🛰️ เชื่อมต่อระบบ Google Earth Engine สำเร็จ!")
except Exception as e:
    print("🔒 กำลังขอเข้าถึงสิทธิ์ผ่าน Terminal...")
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='gee-training-498303')

# ฟังก์ชันแปลงวันที่สุดแปลกของปี 2025 ให้กลายเป็น datetime มาตรฐาน
def clean_pcd_2025_date(row):
    date_str = str(row['Date']).strip()
    year_val = int(row['Year']) if not pd.isna(row['Year']) else 2025
    if date_str in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        return pd.to_datetime(f"1 {date_str} {year_val}")
    try:
        if '-' in date_str:
            parts = date_str.split('-')
            first_part = parts[0].strip()
            second_part = parts[1].strip()
            match_month_year = re.search(r'([A-Za-z]+)\s*(\d{4})?', second_part)
            if match_month_year:
                month_name = match_month_year.group(1)
                return pd.to_datetime(f"{first_part} {month_name} {year_val}")
        return pd.to_datetime(date_str)
    except:
        return pd.to_datetime(f"1 {row['Month']} {year_val}")

# --- โหลดและทำความสะอาดข้อมูลเบื้องต้น ---
df1_raw = pd.read_csv("PCD_Station_WQ_Data_2025.csv")
df1_standard = df1_raw.copy()
df1_standard['Cleaned_Date'] = df1_standard.apply(clean_pcd_2025_date, axis=1)
df1_standard = df1_standard.rename(columns={'Lat': 'station_la', 'Long': 'station_lo', 'tur(NTU)': 'Turbidity_'})
df1_final = df1_standard[['station_la', 'station_lo', 'Cleaned_Date', 'Turbidity_']].rename(columns={'Cleaned_Date': 'Date'})

df2_raw = pd.read_csv("PCD_WQ_Chiangrai_2024.csv")
df2_standard = df2_raw.copy()
df2_standard['Date'] = pd.to_datetime(df2_standard['Date'])
df2_final = df2_standard[['station_la', 'station_lo', 'Date', 'Turbidity_']].copy()

df_cr_all = pd.concat([df1_final, df2_final], axis=0).dropna(subset=['station_la', 'station_lo', 'Date']).reset_index(drop=True)
# กรองค่า Turbidity โมเมออกล่วงหน้า
df_cr_all = df_cr_all[df_cr_all["Turbidity_"] > 0.1].reset_index(drop=True)
print(f"📊 จัดการตารางข้อมูลเรียบร้อย มีข้อมูลทั้งหมด {len(df_cr_all)} แถว ที่ต้องไปดึงค่าดาวเทียม")

# --- ฟังก์ชันจัดการภาพดาวเทียมแบบคลีนเมฆ ---
def maskS2(image):
    qa = image.select("QA60")
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"])

def addIndices(img):
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndti = img.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndssi = img.normalizedDifference(["B4", "B8"]).rename("NDSSI")
    return img.addBands([ndwi, mndwi, ndti, ndssi])

S2_Collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)) # คัดเน้น ๆ เอาภาพที่เมฆน้อยกว่า 50%
      .map(maskS2)
      .map(addIndices))

# ⚡️ สเต็ปความเร็วแสง: วนลูปจับคู่แมตช์ความเร็วสูง
result_data = []
print("\n🚀 กำลังเร่งสปีดสกัดข้อมูลดาวเทียมผ่านคลาวด์ (สเต็ปนี้จะเร็วขึ้นมาก)...")

for index, row in df_cr_all.iterrows():
    try:
        lon, lat = float(row["station_lo"]), float(row["station_la"])
        point = ee.Geometry.Point([lon, lat])
        date = row["Date"]
        
        # ปรับ Filter ค้นหาช่วงภาพดาวเทียมรอบ ๆ วันเก็บตัวอย่างค่าน้ำ
        start_date = (date - timedelta(days=10)).strftime("%Y-%m-%d")
        end_date = (date + timedelta(days=10)).strftime("%Y-%m-%d")
        
        filtered = S2_Collection.filterBounds(point).filterDate(start_date, end_date).sort("CLOUDY_PIXEL_PERCENTAGE")
        
        if filtered.size().getInfo() == 0:
            continue
            
        img = filtered.first()
        # สกัดค่าสถิติจากพิกัดพิกเซลสเกล 10 เมตร
        pixel_values = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10, maxPixels=1e9).getInfo()
        
        if pixel_values is None or pixel_values.get('B2') is None:
            continue
            
        # รวมร่างข้อมูลเดิมและค่าดาวเทียมเข้าด้วยกัน
        combined_row = row.to_dict()
        combined_row['Date'] = date.strftime("%Y-%m-%d")
        combined_row.update(pixel_values)
        result_data.append(combined_row)
        
        if (index + 1) % 50 == 0 or (index + 1) == len(df_cr_all):
            print(f"  -> Progress: สกัดเสร็จแล้ว {index + 1}/{len(df_cr_all)} แถว...")
            
    except Exception as e:
        continue

df_extracted = pd.DataFrame(result_data)
if len(df_extracted) > 0:
    OUTPUT_FILE_CR = "Sentinel2_Extract_ChiangRai.csv"
    df_extracted.to_csv(OUTPUT_FILE_CR, index=False)
    print(f"\n🎉 สำเร็จถล่มทลายครับจี๊ด! ได้ไฟล์เชียงรายฉบับสมบูรณ์ {len(df_extracted)} แถว พร้อมรันโมเดลทำนายแล้วที่: '{OUTPUT_FILE_CR}'")
else:
    print("\n⚠️ ดึงเสร็จแล้วแต่ไม่มีแถวไหนผ่านเงื่อนไขติดเมฆเลย ลองขยายช่วงวันดูไหมครับ?")
    