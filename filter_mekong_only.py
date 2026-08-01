import pandas as pd
import numpy as np

# 1. โหลดไฟล์พิกัดและค่าน้ำดิบของเชียงรายทั้ง 2 ปีเข้ามา (ไฟล์ที่จี๊ดส่งมาให้โมจิ)
try:
    df_2025_raw = pd.read_csv("PCD_Station_WQ_Data_2025.csv")
    df_2024_raw = pd.read_csv("PCD_WQ_Chiangrai_2024.csv")
    print("📋 โหลดตารางน้ำดิบภาคสนามของเชียงรายสำเร็จ!")
except FileNotFoundError as e:
    print(f"❌ ไม่พบไฟล์ตั้งต้น กรุณาตรวจสอบชื่อไฟล์ในโฟลเดอร์น้า: {e}")
    exit()

# 2. ⚡️ สเต็ปกรองมหาเวทย์ "คัดเฉพาะแม่น้ำโขง" ของปี 2025
# (ดูคอลัมน์ 'River' หาแถวที่มีคำว่า Mekong และค่าน้ำปกติไม่เกิน 150 NTU)
df_2025_mekong = df_2025_raw[
    df_2025_raw['River'].str.contains('Mekong', case=False, na=False) & 
    (df_2025_raw['tur(NTU)'] > 0.1) & 
    (df_2025_raw['tur(NTU)'] <= 150)
].copy()

# จัดหน้าตาคอลัมน์แปลงเป็นชื่อมาตรฐานสำหรับปี 2025
df_2025_clean = df_2025_mekong.rename(columns={
    'Lat': 'station_la',
    'Long': 'station_lo',
    'tur(NTU)': 'Turbidity_'
})[['station_la', 'station_lo', 'Date', 'Turbidity_']]
df_2025_clean['Source_Year'] = 2025

# 3. ⚡️ สเต็ปกรอง "คัดเฉพาะแม่น้ำโขง" ของปี 2024 
# (ดูคอลัมน์ 'Surface_Wa' หาแถวที่มีคำว่า Mekong และค่าน้ำปกติไม่เกิน 150 NTU)
df_2024_mekong = df_2024_raw[
    df_2024_raw['Surface_Wa'].str.contains('Mekong', case=False, na=False) & 
    (df_2024_raw['Turbidity_'] > 0.1) & 
    (df_2024_raw['Turbidity_'] <= 150)
].copy()

# จัดหน้าตาคอลัมน์แปลงเป็นชื่อมาตรฐานสำหรับปี 2024
df_2024_clean = df_2024_mekong[['station_la', 'station_lo', 'Date', 'Turbidity_']].copy()
df_2024_clean['Source_Year'] = 2024

# 4. รวมร่างตารางแม่น้ำโขงเชียงรายทั้งสองปีเข้าด้วยกันเป็นตารางเดียว
df_mekong_chiangrai = pd.concat([df_2025_clean, df_2024_clean], axis=0).dropna().reset_index(drop=True)

print("\n📈 ------ สรุปจำนวนข้อมูลแม่น้ำโขง จังหวัดเชียงราย ------")
print(f"🌊 ข้อมูลแม่น้ำโขงปี 2024: {len(df_2024_clean)} สถานี/ช่วงเวลา")
print(f"🌊 ข้อมูลแม่น้ำโขงปี 2025: {len(df_2025_clean)} สถานี/ช่วงเวลา")
print(f"🏆 รวมข้อมูลลุ่มน้ำโขงเชียงรายทั้งหมดหลังคลีนโคลน: {len(df_mekong_chiangrai)} แถว")

# 5. เซฟออกเป็นไฟล์ CSV ใหม่ เพื่อส่งต่อเข้าแล็บสกัดดาวเทียมถัดไป
OUTPUT_NAME = "PCD_Mekong_ChiangRai_Clean.csv"
df_mekong_chiangrai.to_csv(OUTPUT_NAME, index=False)
print(f"\n🎉 คัดกรองและเซฟตารางแม่น้ำโขงให้จี๊ดเรียบร้อยแล้วที่ไฟล์: '{OUTPUT_NAME}'")