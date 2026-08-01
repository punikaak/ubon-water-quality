import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. โหลดข้อมูลและเตรียมฟีเจอร์
df = pd.read_csv("Sentinel2_Extract_Ubon_New.csv")
features = ['B2', 'B3', 'B4', 'B8', 'NDWI', 'MNDWI', 'NDTI', 'NDSSI']
target = 'Turbidity_'

df = df.dropna(subset=features + [target])
df = df[df[target] > 0.0].reset_index(drop=True)

X = df[features]
y = df[target]

# 2. แบ่งข้อมูลเป็นชุด Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ปรับสเกลข้อมูล (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. สร้างและเทรนโมเดล MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='lbfgs',
    max_iter=3000,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 5. ประมวลผลและประเมินผล (Evaluation)
y_pred = mlp.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"--- MLP Evaluation Results ---")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 6. บันทึกโมเดลและ Scaler เก็บไว้ใช้งาน
joblib.dump(mlp, "best_model_neural_network_mlp.pkl")
joblib.dump(scaler, "scaler_ubon.pkl")

# 7. พล็อตกราฟ Validation (Actual vs Predicted)
plt.figure(figsize=(6, 5.5), dpi=150)
sns.set_theme(style="ticks")

plt.scatter(y_test, y_pred, color='#2c7bb6', alpha=0.7, edgecolors='k', s=45, label='Predicted Data')

max_val = max(max(y_test), max(y_pred)) + 5
min_val = min(min(y_test), min(y_pred)) - 5
plt.plot([min_val, max_val], [min_val, max_val], color='#d7191c', linestyle='--', linewidth=1.5, label='1:1 Line')

stats_text = f"Model: Neural Network (MLP)\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f} NTU\n$N$ = {len(y_test)} points"
plt.gca().text(0.05, 0.93, stats_text, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='silver'))

plt.xlabel('Actual Turbidity (NTU)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Turbidity (NTU)', fontsize=11, fontweight='bold')
plt.title('Neural Network (MLP) Model Evaluation', fontsize=12, fontweight='bold')

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig("Plot_Full_Neural_Network_MLP.png", dpi=300)
plt.show()