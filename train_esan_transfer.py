import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# 📂 1. โหลดไฟล์ข้อมูลจากตาราง Calibrate R2 50% ของจี๊ดมาตรง ๆ
df = pd.read_csv("PCD_Esan_To_Ubon_Calibrated_R2_50.csv")

# เลือกคอลัมน์ค่าจริงและคอลัมน์คำตอบของทั้ง 3 โมเดล
models_setup = {
    'Linear Regression': 'Calibrated_Pred_Linear Regression',
    'Random Forest': 'Calibrated_Pred_Random Forest',
    'Neural Network (MLP)': 'Calibrated_Pred_Neural Network (MLP)'
}

print(f"📈 กำลังเริ่มพล็อตกราฟจากข้อมูลดิบทั้งหมด {len(df)} แถว (ครบทุกจุดไม่มีตัดออก)...")

# 🔄 2. วนลูปพล็อตทีละโมเดลจากข้อมูลชุดเดียวกัน
for name, col_name in models_setup.items():
    # ดึงข้อมูลทุกจุดครบ 100% ไม่มีการใช้คำสั่งตัดแถวทิ้ง
    y_actual = df['Turbidity_Actual']
    y_pred = df[col_name]
    
    # คำนวณค่าสถิติจากทุกจุดในตารางนี้ตรง ๆ
    r2_val = r2_score(y_actual, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    # 🎨 3. สั่งพล็อตกราฟตามมาตรฐานเล่มวิจัย
    plt.figure(figsize=(6, 5.5), dpi=150)
    sns.set_theme(style="ticks")
    
    # พล็อตจุดข้อมูลทั้งหมด (จำนวนจุดจะเท่ากับจำนวนแถวใน Excel เป๊ะ)
    plt.scatter(y_actual, y_pred, color='#2c7bb6', alpha=0.7, edgecolors='k', s=45, label='Calibrated Data')
    
    # สร้างเส้นอ้างอิงแนวเฉียง 1:1 Line
    max_val = max(max(y_actual), max(y_pred)) + 10
    min_val = min(min(y_actual), min(y_pred)) - 5
    plt.plot([min_val, max_val], [min_val, max_val], color='#d7191c', linestyle='--', linewidth=1.5, label='1:1 Line')
    
    # แปะกล่องสถิติจริงที่คำนวณได้แบบไม่แต่งแต้มเพิ่ม
    stats_text = f"Model: {name}\n$R^2$ = {r2_val:.3f}\nRMSE = {rmse_val:.3f} NTU\n$N$ = {len(df)} points"
    plt.gca().text(0.05, 0.93, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='silver'))
    
    # กำหนดรายละเอียดแกน
    plt.xlabel('Actual Turbidity (NTU)', fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel('Calibrated Predicted Turbidity (NTU)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title(f'Spatial Validation: {name}\n(Full Dataset, N={len(df)})', fontsize=12, fontweight='bold', pad=12)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # 💾 4. สั่งบันทึกรูปภาพ
    clean_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    output_filename = f"Plot_Full_{clean_name}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f" 🎯 สร้างรูปเสร็จแล้ว -> '{output_filename}' (R2: {r2_val:.4f} | RMSE: {rmse_val:.4f})")

print("\n🏆 บอนรันพล็อตกราฟแบบเก็บครบทุกจุดเสร็จเรียบร้อยแล้วค๊าบจี๊ด!")