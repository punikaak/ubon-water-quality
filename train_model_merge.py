# %% [markdown]
# ### บล็อกที่ 1: รวมร่างข้อมูลปี 2024 และ 2025 เข้าด้วยกัน
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# โหลดข้อมูลทั้งสองปี
df_2024 = pd.read_csv("Sentinel2_Extract.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])
df_2025 = pd.read_csv("Sentinel2_Extract_2025.csv").dropna(subset=['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI'])

# ⭐️ ยุบรวมเข้าด้วยกันเป็นก้อนเดียว เพื่อกระจายตัวเลขให้สมดุล
df_all = pd.concat([df_2024, df_2025], axis=0).reset_index(drop=True)

TARGET_COLUMN = "Turbidity_" 
features = ['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI']

X_raw = df_all[features]
y = df_all[TARGET_COLUMN]

# ปรับสเกลข้อมูลมาตรฐาน
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# สุ่มแบ่งข้อมูล 80% และ 20% จากกองรวมกัน
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 รวมร่างข้อมูลสำเร็จ! มีข้อมูลทั้งหมด {len(df_all)} แถว")
print(f"✅ แบ่งไปเทรน {len(X_train)} แถว และใช้ทดสอบวัดผล {len(X_test)} แถว")

# %% [markdown]
# ### บล็อกที่ 2: รันโมเดลเปรียบเทียบสถิติจากกองข้อมูลรวม
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector (SVR)": SVR(kernel='rbf', C=50, epsilon=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
predictions = {}

print("\n🤖 กำลังเทรนโมเดลจากชุดข้อมูลรวม...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {"R2": r2, "RMSE": rmse}
    predictions[name] = y_pred
    print(f"  -> {name} ประมวลผลเสร็จสิ้น (R²: {r2:.3f})")

df_summary = pd.DataFrame(results).T
print("\n📈 ------ ตารางสรุปความแม่นยำเวอร์ชันรวมข้อมูลสองปี ------")
print(df_summary.to_string())
# %% [markdown]
# ### บล็อกที่ 3: พล็อตกราฟ 4 ช่องเปรียบเทียบ (เวอร์ชันรวมข้อมูลสองปี)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
sns.set_theme(style="ticks")

# หาค่าต่ำสุด-สูงสุดรวม เพื่อให้สเกลเส้นประ 1:1 เท่ากันทุกช่อง
min_val = min(min(y_test), min([np.min(p) for p in predictions.values()]))
max_val = max(max(y_test), max([np.max(p) for p in predictions.values()]))

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    
    # พล็อตจุดข้อมูลจริงแกน X และค่าพยากรณ์แกน Y (ใช้สีน้ำเงินอมม่วงสวย ๆ)
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=50, color='#673ab7', label='Predicted Data')
    
    # ลากเส้นอ้างอิงความแม่นยำ 1:1
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5, label='1:1 Line')
    
    r2_val = results[name]["R2"]
    rmse_val = results[name]["RMSE"]
    stats_text = f'$R^2 = {r2_val:.3f}$\n$RMSE = {rmse_val:.3f}$'
    
    # ฝังกล่องข้อความสถิติ
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_title(name, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Observed Turbidity_ (Field Data)', fontsize=10)
    ax.set_ylabel('Predicted Turbidity_ (Model)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)

# จัดการพื้นที่และเว้นระยะหัวข้อใหญ่ด้านบนอย่างสวยงาม ไม่ทับกันแน่นอน
plt.suptitle('Model Comparison: Merged Data (2024 + 2025)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# เซฟภาพกราฟลงในโฟลเดอร์ถาวร
plt.savefig("Model_Comparison_Merged_Grid.png", dpi=300)
plt.show()
print("\n🎉 พล็อตกราฟและเซฟไฟล์ 'Model_Comparison_Merged_Grid.png' เรียบร้อยแล้วครับจี๊ด!")