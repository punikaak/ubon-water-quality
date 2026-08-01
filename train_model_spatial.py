# %% [markdown]
# ### บล็อกที่ 1: โหลดข้อมูลและกรองค่าความขุ่นที่เป็น 0 ออก (ฉบับแก้ไขค่าเอ๋อ)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# 1. โหลดข้อมูลอุบลฯ (ชุดเทรน)
df_ub_2024 = pd.read_csv("Sentinel2_Extract.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_ub_2025 = pd.read_csv("Sentinel2_Extract_2025.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_train_base = pd.concat([df_ub_2024, df_ub_2025], axis=0).reset_index(drop=True)

# 2. โหลดข้อมูลสกลนคร (ชุดทดสอบ) พร้อมจัดการ Drop แถวที่ค่าว่างออก
df_spatial_test = pd.read_csv("Sentinel2_Extract_Sakon.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])

TARGET_COLUMN = "Turbidity_" 

# ⭐️ ทริกเด็ด: บังคับเลือกเฉพาะแถวที่ค่าความขุ่นมากกว่า 0 จริง ๆ (ตัดค่าวัดปลอม/ค่าเอ๋อออก)
df_spatial_test = df_spatial_test[df_spatial_test[TARGET_COLUMN] > 0.1].reset_index(drop=True)

features = ['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI']

X_train_raw = df_train_base[features]
y_train = df_train_base[TARGET_COLUMN]

X_test_raw = df_spatial_test[features]
y_test = df_spatial_test[TARGET_COLUMN]

# ปรับสเกลข้อมูลดาวเทียม
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"📊 โหลดตารางและกรองค่าเอ๋อเรียบร้อย!")
print(f"🌲 ข้อมูลสำหรับเทรน (อุบลฯ): {len(df_train_base)} แถว")
print(f"🚀 ข้อมูลสำหรับทดสอบพยากรณ์จริง (สกลนคร คัดเหลือค่าน้ำจริง): {len(df_spatial_test)} แถว")

# %% [markdown]
# ### บล็อกที่ 2: สั่งสอนสมodel แล้วส่งข้ามถิ่นไปทำนายสกลนคร
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector (SVR)": SVR(kernel='rbf', C=50, epsilon=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
predictions = {}

print("\n🤖 กำลังส่งโมเดลข้ามถิ่นไปพยากรณ์พื้นที่สกลนคร...")
for name, model in models.items():
    model.fit(X_train, y_train)       # สอนโมเดลด้วยข้อมูลอุบลฯ
    y_pred = model.predict(X_test)    # ทายค่าน้ำของสกลนครจากค่าดาวเทียม
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {"Spatial_R2": r2, "Spatial_RMSE": rmse}
    predictions[name] = y_pred
    print(f"  -> {name} ทำนายเสร็จสิ้น (Spatial R²: {r2:.3f})")

df_summary = pd.DataFrame(results).T
print("\n📈 ------ ตารางสรุปผลความแม่นยำการทดสอบข้ามพื้นที่ (สกลนคร) ------")
print(df_summary.to_string())

# %% [markdown]
# ### บล็อกที่ 3: พล็อตกราฟ 4 ช่องชนกัน (Spatial Validation)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
sns.set_theme(style="ticks")

min_val = min(min(y_test), min([np.min(p) for p in predictions.values()]))
max_val = max(max(y_test), max([np.max(p) for p in predictions.values()]))

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    
    # พล็อตจุดข้อมูลจริงสกลนครแกน X และค่าที่โมเดลเดาได้แกน Y (ใช้สีส้มอิฐสไตล์แม่น้ำโขง)
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=50, color='#d35400', label='Sakon Predicted')
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')
    
    r2_val = results[name]["Spatial_R2"]
    rmse_val = results[name]["Spatial_RMSE"]
    stats_text = f'$Spatial R^2 = {r2_val:.3f}$\n$RMSE = {rmse_val:.3f}$'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_title(name, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Observed Turbidity_ (Actual Sakon Data)', fontsize=10)
    ax.set_ylabel('Predicted Turbidity_ (Trained on Ubon)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)

plt.suptitle('Spatial Model Validation: Trained on Ubon, Tested on Sakon/Mekong', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# เซฟภาพกราฟไปเคลมในบทที่ 4 หัวข้อแบบจำลองระดับภูมิภาค
plt.savefig("Model_Spatial_Validation_Grid.png", dpi=300)
plt.show()
print("\n🎉 พล็อตกราฟ Spatial Validation และเซฟไฟล์ 'Model_Spatial_Validation_Grid.png' เรียบร้อยครับจี๊ด!")