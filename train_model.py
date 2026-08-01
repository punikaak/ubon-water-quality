# %% [markdown]
# ### บล็อกที่ 1: โหลดข้อมูลและเพิ่มตัวปรับสเกล (StandardScaler)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# ⭐️ นำเข้าตัวปรับสเกลข้อมูลมาตรฐาน
from sklearn.preprocessing import StandardScaler

# โหลดตาราง
df_2024 = pd.read_csv("Sentinel2_Extract.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_2025 = pd.read_csv("Sentinel2_Extract_2025.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])

TARGET_COLUMN = "Turbidity_" 
features = ['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI']

X_train_raw = df_2024[features]
y_train = df_2024[TARGET_COLUMN]

X_val2025_raw = df_2025[features]
y_val2025 = df_2025[TARGET_COLUMN]

# ⭐️ ทำการแปลงสเกลข้อมูลดาวเทียมของทั้งสองปีให้เป็นมาตรฐานเดียวกัน
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)       # ปรับฐานด้วยปี 2024
X_val2025 = scaler.transform(X_val2025_raw)      # ปรับปี 2025 ให้เข้าเกณฑ์เดียวกัน

print(f"📊 ปรับสเกลข้อมูลดาวเทียมสำเร็จ! พร้อมรันโมเดลใหม่อีกครั้ง")

# %% [markdown]
# ### บล็อกที่ 2: รันโมเดลฉบับปรับสเกลแล้วพ่นสถิติ
models = {
    "Linear Regression": LinearRegression(),
    # ปรับจูน SVR ให้ยืดหยุ่นขึ้นเข้ากับสเกลใหม่
    "Support Vector (SVR)": SVR(kernel='rbf', C=10, epsilon=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
predictions = {}

print("\n🤖 กำลังประมวลผลโมเดลเวอร์ชันปรับสเกล...")
for name, model in models.items():
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_val2025) 
    
    r2 = r2_score(y_val2025, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val2025, y_pred))
    
    results[name] = {"R2_2025": r2, "RMSE_2025": rmse}
    predictions[name] = y_pred
    print(f"  -> {name} เสร็จสิ้น (R² ใหม่: {r2:.3f})")

df_summary = pd.DataFrame(results).T
print("\n📈 ------ ตารางสรุปความแม่นยำหลังปรับสเกล ------")
print(df_summary.to_string())