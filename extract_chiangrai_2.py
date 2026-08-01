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
    
    # เคสที่ 1: มาแค่ชื่อเดือนทื่อ ๆ เช่น 'Mar', 'Apr'
    if date_str in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        return pd.to_datetime(f"1 {date_str} {year_val}")
        
    # เคสที่ 2: มาเป็นช่วง เช่น '13-16 May 2025' หรือ '30 Jun-4 Jul 2025'
    try:
        # ตัดเอาเฉพาะข้อความหลังเครื่องหมายขีด (-) ออกมา เพื่อหาวันและเดือนหลัก
        if '-' in date_str:
            parts = date_str.split('-')
            first_part = parts[0].strip() # ดึงเลขวันตัวแรกมา เช่น 13
            second_part = parts[1].strip() # ส่วนหลัง เช่น '16 May 2025' หรือ '4 Jul 2025'
            
            # สกัดเอาชื่อเดือนและปีจากส่วนหลัง
            match_month_year = re.search(r'([A-Za-z]+)\s*(\d{4})?', second_part)
            if match_month_year:
                month_name = match_month_year.group(1)
                # รวมร่างเป็น วันแรกของช่วง + เดือน + ปี
                return pd.to_datetime(f"{first_part} {month_name} {year_val}")
        
        # เคสปกติที่อาจจะหลุดมา
        return pd.to_datetime(date_str)
    except:
        # ถ้าแปลงไม่สำเร็จจริง ๆ ให้ล็อกวันที่ 1 ของเดือนตามคอลัมน์ Month
        return pd.to_datetime(f"1 {row['Month']} {year_val}")

# --- สตาร์ทโหลดข้อมูล ---
# 🚨 โหลดและปรับโครงสร้างไฟล์ชุดที่ 1 (2025)
df1_raw = pd.read_csv("PCD_Station_WQ_Data_2025.csv")
df1_standard = df1_raw.copy()
# ส่งไปคลีนวันที่ทีละแถว
df1_standard['Cleaned_Date'] = df1_standard.apply(clean_pcd_2025_date, axis=1)
df1_standard = df1_standard.rename(columns={
    'Lat': 'station_la',
    'Long': 'station_lo',
    'tur(NTU)': 'Turbidity_'
})
df1_final = df1_standard[['station_la', 'station_lo', 'Cleaned_Date', 'Turbidity_']].rename(columns={'Cleaned_Date': 'Date'})

# 🚨 โหลดและปรับโครงสร้างไฟล์ชุดที่ 2 (2024)
df2_raw = pd.read_csv("PCD_Mekong_ChiangRai_Clean.csv")
df2_standard = df2_raw.copy()
df2_standard['Date'] = pd.to_datetime(df2_standard['Date'])
df2_final = df2_standard[['station_la', 'station_lo', 'Date', 'Turbidity_']].copy()

# มัดรวมตารางเชียงรายทั้งสองปีเข้าด้วยกันเป็นโครงสร้างมาตรฐานเดียวกันเป๊ะ
df_cr_all = pd.concat([df1_final, df2_final], axis=0).dropna(subset=['station_la', 'station_lo', 'Date']).reset_index(drop=True)
print(f"📊 ปรับตารางและล้างบั๊กวันที่สำเร็จ! มีข้อมูลพร้อมสกัดดาวเทียมทั้งหมด {len(df_cr_all)} แถว")

# --- วิ่งสกัด Google Earth Engine ---
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

S2_Collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 100))
      .map(maskS2)
      .map(addIndices))

result_data = []
print("\n🚀 เริ่มสกัดข้อมูลดาวเทียมเชียงราย...")

for index, row in df_cr_all.iterrows():
    try:
        lon = float(row["station_lo"])
        lat = float(row["station_la"])
        point = ee.Geometry.Point([lon, lat])

        date = row["Date"]
        start = (date - timedelta(days=15)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=15)).strftime("%Y-%m-%d")

        filtered = S2_Collection.filterBounds(point).filterDate(start, end).sort("CLOUDY_PIXEL_PERCENTAGE")
        if filtered.size().getInfo() == 0:
            continue

        img = filtered.first()
        value = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10, maxPixels=1e13).getInfo()

        if value is None or value.get('B2') is None:
            continue

        row_data = row.to_dict()
        row_data['Date'] = date.strftime("%Y-%m-%d") # แปลงกลับเป็น string สวย ๆ ตอนเซฟไฟล์
        row_data.update(value)
        result_data.append(row_data)
        if (index+1) % 10 == 0 or (index+1) == len(df_cr_all):
            print(f"  -> ดำเนินการถึงแถวที่ {index+1}/{len(df_cr_all)}...")
    except Exception as e:
        continue

df_extracted = pd.DataFrame(result_data)
if len(df_extracted) > 0:
    df_extracted = df_extracted[df_extracted["Turbidity_"] > 0.1].reset_index(drop=True)
    OUTPUT_FILE_CR = "Sentinel2_Extract_ChiangRai_2.csv"
    df_extracted.to_csv(OUTPUT_FILE_CR, index=False)
    print(f"\n🎉 ล้างบั๊กวันที่และสกัดข้อมูลดาวเทียมเสร็จสิ้นค๊าบ! ได้ไฟล์: '{OUTPUT_FILE_CR}'")