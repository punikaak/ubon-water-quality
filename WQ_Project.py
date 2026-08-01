import pandas as pd
import numpy as np
from datetime import timedelta
import ee

# ========================================================
# 1. เชื่อมต่อระบบดาวเทียม Google Earth Engine
# ========================================================
try:
    ee.Initialize(project='gee-training-498303')
    print("🛰️ เชื่อมต่อระบบ Google Earth Engine สำเร็จ!")
except Exception as e:
    print("🔒 กำลังขอเข้าถึงสิทธิ์ผ่าน Terminal...")
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='gee-training-498303')

# ========================================================
# 2. อ่านไฟล์ข้อมูลภาคสนาม PCD ของจี๊ด
# ========================================================
INPUT_FILE = "PCD_WQ_Ubon_2024_Table.csv" 

try:
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"📊 โหลดไฟล์ PCD สำเร็จ! พบข้อมูลทั้งหมด {len(df_raw)} แถว")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์: {INPUT_FILE} ในโฟลเดอร์ D:\\Chula\\ADPC")
    print("👉 จี๊ดเช็กดูอีกทีน้าว่าไฟล์นี้เซฟอยู่ในโฟลเดอร์เดียวกันกับสคริปต์โค้ดหรือยัง")
    exit()

# ========================================================
# 3. ฟังก์ชันจัดการเมฆ (เวอร์ชัน .And ตัวใหญ่ ถูกไวยากรณ์ GEE ชัวร์)
# ========================================================
def maskS2(image):
    qa = image.select("QA60")
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    
    no_cloud = qa.bitwiseAnd(cloudBitMask).eq(0)
    no_cirrus = qa.bitwiseAnd(cirrusBitMask).eq(0)
    
    # ใช้ .And() เพื่อสกัดเงื่อนไขระดับพิกเซลของคลังภาพดาวเทียม
    mask = no_cloud.And(no_cirrus)
    
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"])

def addIndices(img):
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndti = img.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndssi = img.normalizedDifference(["B4", "B8"]).rename("NDSSI")
    return img.addBands([ndwi, mndwi, ndti, ndssi])

# เรียกคลังภาพ Sentinel-2 ของปี 2024
S2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterDate("2024-01-01", "2024-12-31")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 100))
      .map(maskS2)
      .map(addIndices))

# ========================================================
# 4. ลูปยิงพิกัดสถานีไปสกัดดึงข้อมูลจาก GEE
# ========================================================
result_data = []
print("\n🚀 เริ่มกระบวนการสกัดข้อมูลดาวเทียมลงตาราง PCD...")

for index, row in df_raw.iterrows():
    try:
        lon = float(row["station_lo"])
        lat = float(row["station_la"])
        point = ee.Geometry.Point([lon, lat])

        date = pd.to_datetime(row["Date"])
        start = (date - timedelta(days=15)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=15)).strftime("%Y-%m-%d")

        filtered = S2.filterBounds(point).filterDate(start, end).sort("CLOUDY_PIXEL_PERCENTAGE")
        if filtered.size().getInfo() == 0:
            print(f"  -> แถว {index+1}: ❌ ไม่พบรอบวงโคจรช่วงนี้")
            continue

        img = filtered.first()
        
        value = img.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=point, 
            scale=10,
            maxPixels=1e13
        ).getInfo()

        if value is None or value.get('B2') is None:
            print(f"  -> แถว {index+1}: ☁️ ข้ามพิกเซลที่ติดเมฆทึบ")
            continue

        row_data = row.to_dict()
        row_data.update(value)
        result_data.append(row_data)
        print(f"  -> แถว {index+1}: ✅ สกัดข้อมูลดาวเทียมสำเร็จ!")

    except Exception as e:
        print(f"  -> แถว {index+1}: ⚠️ เกิดปัญหา: {e}")
        continue

# ========================================================
# 5. บันทึกผลลัพธ์เป็นตารางใหม่ลงเครื่องคอมพิวเตอร์
# ========================================================
df_extracted = pd.DataFrame(result_data)
if len(df_extracted) > 0:
    OUTPUT_FILE = "Sentinel2_Extract.csv"
    df_extracted.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎉 เสร็จสิ้นภารกิจ! สร้างไฟล์ตารางใหม่เรียบร้อย: '{OUTPUT_FILE}'")
    print(f"📊 ได้รับข้อมูลรวม {len(df_extracted)} แถว พร้อมเอาไปใช้รันโมเดลแล้วครับจี๊ด!")
else:
    print("\n⚠️ รันเสร็จแต่ได้ผลลัพธ์ 0 แถว ให้เช็กหัวชื่อคอลัมน์พิกัดในไฟล์ PCD ดูอีกครั้งน้า")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ========================================================
# 1. โหลดข้อมูลที่เพิ่งสกัดค่าดาวเทียมมาได้
# ========================================================
INPUT_FILE = "Sentinel2_Extract.csv"

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"📊 โหลดข้อมูลสำเร็จ! จำนวนทั้งหมด {len(df)} แถว")
except FileNotFoundError:
    print(f"❌ หาไฟล์ '{INPUT_FILE}' ไม่เจอ! จี๊ดตรวจสอบให้ชัวร์ว่ารันโค้ดตัวแรกเสร็จและมีไฟล์นี้อยู่ในโฟลเดอร์แล้วน้า")
    exit()

# ลบแถวที่มีค่าว่าง (NaN) ในคอลัมน์ดาวเทียมออกเพื่อป้องกันโมเดลเอ๋อ
df = df.dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSi'])

# ========================================================
# 2. 🚨 จุดสำคัญ: เลือกคอลัมน์ "ค่าภาคสนาม PCD" ที่ต้องการพยากรณ์
# ========================================================
# 📌 จี๊ดตรวจสอบชื่อคอลัมน์ค่าภาคสนามในตารางของจี๊ดน้า (เช่น 'Turbidity', 'DO', 'SS' หรือ 'TSS') 
# ให้เปลี่ยนตัวหนังสือในเครื่องหมายคำพูดข้างล่างนี้ให้ตรงกับชื่อคอลัมน์ในตารางจริง
TARGET_COLUMN = "Turbidity" 

if TARGET_COLUMN not in df.columns:
    print(f"❌ ไม่พบคอลัมน์ '{TARGET_COLUMN}' ในไฟล์ CSV ของคุณ!")
    print(f"👉 คอลัมน์ที่มีให้เลือกในไฟล์ของจี๊ดคือ: {list(df.columns)}")
    print("แก้ไขชื่อคอลัมน์ที่ตัวแปร TARGET_COLUMN ในโค้ดแล้วลองรันใหม่อีกครั้งนะครับ")
    exit()

# ========================================================
# 3. เตรียมข้อมูลคุณลักษณะ (Features) และค่าเป้าหมาย (Target)
# ========================================================
# ดึงค่าแบนด์สะท้อนแสงหลัก และดัชนีน้ำที่คำนวณมามาใช้เป็นปัจจัยตัวแปรต้น
features = ['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSi']

X = df[features]
y = df[TARGET_COLUMN]

# แบ่งข้อมูลเป็นเซต Train 80% สำหรับสอนโมเดล และ Test 20% สำหรับประเมินความแม่นยำ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================================
# 4. สร้างและเทรนโมเดล Random Forest
# ========================================================
print(f"🤖 กำลังเทรนโมเดล Random Forest Regressor เพื่อพยากรณ์ค่า {TARGET_COLUMN}...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ทำนายผลข้อมูลชุดทดสอบ (Test Set)
y_pred = model.predict(X_test)

# คำนวณดัชนีวัดผลทางสถิติ (R2 Score และ RMSE)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📈 --- ผลการวัดความแม่นยำของโมเดล ---")
print(f"✨ ค่า R-squared (R²): {r2:.3f}")
print(f"🎯 ค่า RMSE: {rmse:.3f}")

# ========================================================
# 5. พล็อตภาพกราฟ Scatter Plot และเซฟลงคอมพิวเตอร์
# ========================================================
plt.figure(figsize=(8, 6))
sns.set_theme(style="ticks")

# พล็อตจุดไข่ปลาเปรียบเทียบค่าจริงกับค่าทำนาย
plt.scatter(y_test, y_pred, color='#2ca02c', alpha=0.7, edgecolors='k', s=50, label='Predicted Data')

# ลากเส้นอ้างอิง 1:1 (Perfect Prediction Line)
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit (1:1)')

# ตกแต่งหน้าตากราฟให้สวยงาม Scannable อ่านง่ายสำหรับใส่เล่มวิจัย
plt.title(f'Water Quality Prediction: Random Forest Model ({TARGET_COLUMN})', fontsize=14, pad=15, fontweight='bold')
plt.xlabel(f'Observed {TARGET_COLUMN} (PCD Field Data)', fontsize=12, labelpad=10)
plt.ylabel(f'Predicted {TARGET_COLUMN} (Sentinel-2 Derived)', fontsize=12, labelpad=10)

# แสดงข้อความสถิติ R2 และ RMSE บนหน้าจอกราฟ
stats_text = f'$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$'
plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# เซฟรูปกราฟลงในโฟลเดอร์คอมถาวร
OUTPUT_IMAGE = f"Prediction_Plot_{TARGET_COLUMN}.png"
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.close()

print(f"\n🎉 สำเร็จแล้วจี๊ด! โมเดลรันเสร็จเรียบร้อยและสร้างไฟล์รูปกราฟสถิติให้แล้วที่: '{OUTPUT_IMAGE}'")