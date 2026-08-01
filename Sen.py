import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. โหลดข้อมูลจากไฟล์ CSV ของคุณ
df = pd.read_csv('Sentinel2_Extract copy - Copy.csv')

# 2. ทำความสะอาดข้อมูลเบื้องต้น (ลบแถวที่มีค่าว่าง)
df = df.dropna()

# 3. กำหนดตัวแปรต้น (X) และตัวแปรตาม (y)
# การใช้ดรอปคอลัมน์ NTU ออก หมายถึงเราใช้ทุกตัวแปรที่เหลือเพื่อดัน R^2 ให้สูงสุด
X = df.drop(columns=['NTU'])
y = df['NTU']

# 4. สร้างและเทรนโมเดล Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)

# 5. คำนวณค่า R-squared
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# 6. พิมพ์ผลลัพธ์สมการที่นำไปใช้ได้เลย
print(f"ค่า R-squared สูงสุดที่ได้: {r2:.4f}\n")
print("สมการที่ได้คือ:")
print(f"NTU = {model.intercept_:.4f}", end="")

for i, col in enumerate(X.columns):
    coef = model.coef_[i]
    if coef >= 0:
        print(f" + {coef:.4f}({col})", end="")
    else:
        print(f" - {abs(coef):.4f}({col})", end="")
print()