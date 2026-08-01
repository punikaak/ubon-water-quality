# %% [markdown]
# ### บล็อกที่ 1: โหลดข้อมูลและกรองเฉพาะข้อมูลใน "แม่น้ำโขง" ช่วงค่าน้ำปกติ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. โหลดข้อมูลฐานภาคอีสาน (Train บนบริบทแม่น้ำโขงตอนล่าง)
df_ub24 = pd.read_csv("Sentinel2_Extract.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_ub25 = pd.read_csv("Sentinel2_Extract_2025.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_sk24 = pd.read_csv("Sentinel2_Extract_Sakon.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_train_base = pd.concat([df_ub24, df_ub25, df_sk24], axis=0).reset_index(drop=True)

# 2. โหลดข้อมูลสกัดดาวเทียมของฝั่งเชียงราย (Test)
df_chiangrai = pd.read_csv("Sentinel2_Extract_ChiangRai_2.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])

TARGET_COLUMN = "Turbidity_" 
features = ['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI']

# ⚡️ กรองฝั่ง Train: คัดกรองช่วงค่าน้ำปกติ (ไม่เกิน 150 NTU)
df_train_base = df_train_base[(df_train_base[TARGET_COLUMN] > 0.1) & (df_train_base[TARGET_COLUMN] <= 150)].reset_index(drop=True)

# ⚡️ กรองฝั่ง Test (เชียงราย): คัดเอาเฉพาะสถานีตรวจวัดที่อยู่ในแม่น้ำโขง และค่าน้ำไม่เกิน 150 NTU
# ตรวจสอบชื่อคอลัมน์ระบบลุ่มน้ำ (ดักกรณีใช้ชื่อคอลัมน์ Surface_Wa หรือ River)
river_col = 'Surface_Wa' if 'Surface_Wa' in df_chiangrai.columns else ('River' if 'River' in df_chiangrai.columns else None)

if river_col:
    df_chiangrai = df_chiangrai[df_chiangrai[river_col].str.contains('Mekong', case=False, na=False)]
else:
    print("⚠️ คำเตือน: ไม่พบคอลัมน์ระบุชื่อแม่น้ำในไฟล์ดักข้อมูล")

df_chiangrai = df_chiangrai[(df_chiangrai[TARGET_COLUMN] > 0.1) & (df_chiangrai[TARGET_COLUMN] <= 150)].reset_index(drop=True)

X_train_raw = df_train_base[features]
y_train = df_train_base[TARGET_COLUMN]

X_test_raw = df_chiangrai[features]
y_test = df_chiangrai[TARGET_COLUMN]

# ปรับสเกลข้อมูลดาวเทียมด้วย StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"🌊 ล็อกพิกัดลุ่มน้ำโขงเชียงรายสำเร็จ!")
print(f"🌲 ข้อมูลสอน (แม่น้ำโขงภาคอีสาน): {len(df_train_base)} แถว")
print(f"🏔️ ข้อมูลทดสอบ (แม่น้ำโขงเชียงรายเท่านั้น): {len(df_chiangrai)} แถว")

# %% [markdown]
# ### บล็อกที่ 2: รันทำนายผลเปรียบเทียบสถิติโมเดลแม่น้ำโขง
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
}

results = {}
predictions = {}

print("\n🤖 กำลังส่งโมเดลข้ามภูมิภาค (แม่น้ำโขง ชน แม่น้ำโขง)...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {"Mekong_Only_R2": r2, "Mekong_Only_RMSE": rmse}
    predictions[name] = y_pred
    print(f"  -> {name} พยากรณ์เสร็จแล้ว")

df_summary = pd.DataFrame(results).T
print("\n📈 ------ ตารางสรุปสถิติความแม่นยำ (เฉพาะแม่น้ำโขงเชียงราย) ------")
print(df_summary.to_string())

# %% [markdown]
# ### บล็อกที่ 3: พล็อตกราฟเปรียบเทียบผลลัพธ์ลุ่มน้ำโขงล้วน
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
sns.set_theme(style="ticks")

min_val = min(min(y_test), min([np.min(p) for p in predictions.values()]))
max_val = max(max(y_test), max([np.max(p) for p in predictions.values()]))

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    
    # ใช้สีส้มอิฐอมน้ำตาลสะท้อนเฉดสีดินตะกอนแม่น้ำโขง
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=55, color='#d35400', label='Mekong Test Point')
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5, label='1:1 Line')
    
    r2_val = results[name]["Mekong_Only_R2"]
    rmse_val = results[name]["Mekong_Only_RMSE"]
    stats_text = f'$R^2 = {r2_val:.3f}$\n$RMSE = {rmse_val:.3f}$'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_title(name, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Observed Turbidity_ (Mekong Actual)', fontsize=10)
    ax.set_ylabel('Predicted Turbidity_ (Model Forecast)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)

plt.suptitle('Cross-Regional Spatial Validation: Restricted to Mekong River System', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

plt.savefig("Model_Mekong_Only_Validation_Grid.png", dpi=300)
plt.show()
print("\n🎉 เซฟกราฟลุ่มน้ำโขงลงไฟล์ 'Model_Mekong_Only_Validation_Grid.png' เรียบร้อยครับ!")
# %% [markdown]
# ### บล็อกที่ 4: สร้างตารางข้อมูลดิบเปรียบเทียบรายแถว (Actual vs Predicted)
# สร้าง DataFrame กองกลางเพื่อรวมคำตอบ
df_prediction_table = pd.DataFrame({
    'Date': df_chiangrai['Date'].dt.strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(df_chiangrai['Date']) else df_chiangrai['Date'],
    'Turbidity_Actual': y_test.values
})

# ลูปดึงค่าที่แต่ละโมเดลทายได้ มาเสียบเป็นคอลัมน์ใหม่
for name, y_pred in predictions.items():
    df_prediction_table[f'Pred_{name}'] = y_pred

# ทำการปัดเศษทศนิยมให้เหลือ 3 ตำแหน่งเพื่อให้ดูง่ายและสะอาดตาเวลาลงเล่ม
df_prediction_table = df_prediction_table.round(3)

# 1. ปริ๊นท์ตัวอย่าง 15 แถวแรกขึ้นมาส่องบนหน้าจอคอนโซล
print("\n📋 ------ ตัวอย่างตารางข้อมูลดิบหลังเข้าโมเดล (15 แถวแรก) ------")
print(df_prediction_table.head(15).to_string(index=False))

# 2. 💾 เซฟตารางทั้งหมดออกเป็นไฟล์ CSV เพื่อเอาไปเปิดดู/ก๊อปปี้ใน Excel ได้ง่าย ๆ
OUTPUT_TABLE_NAME = "ChiangRai_Mekong_Prediction_Results.csv"
df_prediction_table.to_csv(OUTPUT_TABLE_NAME, index=False)

print(f"\n🎉 เซฟตารางผลลัพธ์รายแถวสำเร็จแล้วครับจี๊ด! ไฟล์อยู่ที่: '{OUTPUT_TABLE_NAME}'")
print(f"👉 จี๊ดสามารถดับเบิ้ลคลิกเปิดไฟล์นี้ใน Excel เพื่อก๊อปตารางไปใส่บทที่ 4 ได้เลยน้า")