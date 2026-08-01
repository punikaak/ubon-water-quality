# %% [markdown]
# ### บล็อกที่ 1 (เวอร์ชันคัดกรองขอบเขตค่าน้ำลุ่มน้ำเหนือ): โหลดและคัดค่าน้ำหลุดโลกออก
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. โหลดข้อมูลฐานภาคอีสาน (อุบลฯ + สกลนคร)
df_ub24 = pd.read_csv("Sentinel2_Extract.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_ub25 = pd.read_csv("Sentinel2_Extract_2025.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_sk24 = pd.read_csv("Sentinel2_Extract_Sakon.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_train_base = pd.concat([df_ub24, df_ub25, df_sk24], axis=0).reset_index(drop=True)

# 2. โหลดข้อมูลเชียงราย
df_chiangrai = pd.read_csv("Sentinel2_Extract_ChiangRai.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])

TARGET_COLUMN = "Turbidity_" 

# ⭐️ ทริกแก้วิกฤต: บังคับจำกัดสเกลค่าน้ำทั้งอีสานและเชียงรายให้อยู่ในช่วงปกติ (ไม่เกิน 150 NTU)
# เพื่อตัดค่าน้ำโคลนหลาก 3,000 NTU ที่ดาวเทียมวัดไม่ได้ออกไปให้หมด
df_train_base = df_train_base[(df_train_base[TARGET_COLUMN] > 0.1) & (df_train_base[TARGET_COLUMN] <= 150)].reset_index(drop=True)
df_chiangrai = df_chiangrai[(df_chiangrai[TARGET_COLUMN] > 0.1) & (df_chiangrai[TARGET_COLUMN] <= 150)].reset_index(drop=True)

features = ['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI']

X_train_raw = df_train_base[features]
y_train = df_train_base[TARGET_COLUMN]

X_test_raw = df_chiangrai[features]
y_test = df_chiangrai[TARGET_COLUMN]

# ปรับสเกลข้อมูลดาวเทียม
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"📊 คลีนข้อมูลและตัดค่าน้ำโคลนไหลหลากออกสำเร็จ!")
print(f"🌲 ข้อมูลสอนโมเดลช่วงปกติ (ภาคอีสาน): {len(df_train_base)} แถว")
print(f"🏔️ ข้อมูลทดสอบพยากรณ์ช่วงปกติ (เชียงราย): {len(df_chiangrai)} แถว")
# %% [markdown]
# ### บล็อกที่ 2: ให้โมเดลลองทายค่าน้ำเชียงราย และกางตารางดูคะแนน
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
}

results = {}
predictions = {}

print("\n🤖 กำลังส่งโมเดลจากอีสานขึ้นเหนือไปทายค่าน้ำที่เชียงราย...")
for name, model in models.items():
    model.fit(X_train, y_train)    # สอนโมเดลด้วยบริบทน้ำภาคอีสาน
    y_pred = model.predict(X_test) # ให้เดาคำตอบค่าน้ำเชียงรายจากเฉดสีดาวเทียม
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {"ChiangRai_R2": r2, "ChiangRai_RMSE": rmse}
    predictions[name] = y_pred
    print(f"  -> {name} พยากรณ์เชียงรายเสร็จแล้ว (R²: {r2:.3f})")

df_summary = pd.DataFrame(results).T
print("\n📈 ------ ตารางสรุปสถิติความแม่นยำ ณ จังหวัดเชียงราย ------")
print(df_summary.to_string())

# %% [markdown]
# ### บล็อกที่ 3: พล็อตกราฟเปรียบเทียบพยากรณ์เชียงราย
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
sns.set_theme(style="ticks")

min_val = min(min(y_test), min([np.min(p) for p in predictions.values()]))
max_val = max(max(y_test), max([np.max(p) for p in predictions.values()]))

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    
    # พล็อตจุดจริงแกน X จุดทายแกน Y (ใช้สีเขียวหัวเป็ดสไตล์ลุ่มน้ำภาคเหนือ)
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=45, color='#117a65', label='Predicted WQ')
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5, label='1:1 Line')
    
    r2_val = results[name]["ChiangRai_R2"]
    rmse_val = results[name]["ChiangRai_RMSE"]
    stats_text = f'$R^2 = {r2_val:.3f}$\n$RMSE = {rmse_val:.3f}$'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_title(name, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Observed Turbidity_ (ChiangRai Actual)', fontsize=10)
    ax.set_ylabel('Predicted Turbidity_ (Model Forecast)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)

plt.suptitle('Cross-Regional Model Validation: Trained on Esan, Tested on Chiang Rai', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

# เซฟรูปเก็บไว้เป็นไม้เด็ดในบทที่ 4 สำหรับหัวข้อการขยายผลทดสอบข้ามภูมิภาค
plt.savefig("Model_ChiangRai_Validation_Grid.png", dpi=300)
plt.show()
print("\n🎉 พล็อตกราฟเชียงรายและเซฟไฟล์ 'Model_ChiangRai_Validation_Grid.png' เรียบร้อยครับจี๊ด!")