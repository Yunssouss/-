import pandas as pd

# إنشاء DataFrame يحتوي على معطيات الإنتاج اليومي للطاقة الشمسية والريحية لشهر 11 من 2023
data = pd.DataFrame({
    'التاريخ': pd.date_range(start='2023-11-01', end='2023-11-30', freq='D'),
    'إنتاج الطاقة الشمسية (kW)': [500, 520, 530, 510, 495, 480, 470, 490, 510, 520, 530, 540, 550, 530, 510, 520, 500, 490, 495, 510, 520, 530, 545, 555, 560, 570, 565, 550, 540, 530],
    'إنتاج الطاقة الريحية (kW)': [300, 290, 310, 320, 305, 300, 295, 310, 325, 335, 340, 320, 310, 300, 305, 315, 330, 340, 335, 325, 315, 310, 305, 320, 330, 340, 345, 350, 340, 330],
    'سرعة الرياح (كم/ساعة)': [20, 18, 22, 25, 21, 19, 17, 22, 24, 26, 23, 21, 20, 19, 22, 25, 27, 29, 26, 23, 21, 20, 19, 22, 25, 28, 27, 26, 24, 23],
    'الإشعاع الشمسي (W/m²)': [600, 620, 630, 610, 595, 580, 570, 590, 610, 620, 630, 640, 650, 630, 610, 620, 600, 590, 595, 610, 620, 630, 645, 655, 660, 670, 665, 650, 640, 630],
    'درجة الحرارة (°C)': [25, 23, 22, 24, 26, 25, 24, 23, 22, 25, 26, 27, 28, 26, 25, 24, 23, 22, 24, 25, 26, 27, 28, 29, 30, 31, 29, 28, 26, 25]
})

# عرض المعطيات الأولية
data.head()


from sklearn.model_selection import train_test_split

# تقسيم المعطيات
X = data[['سرعة الرياح', 'الإشعاع الشمسي', 'درجة الحرارة']]  # الخصائص المستعملة
y = data['إنتاج الطاقة الشمسية']  # القيمة اللي بغينا نتنبؤو بها (أو نقدر نستعمل إنتاج الطاقة الريحية)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

# إعداد النموذج
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# تدريب النموذج
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

from sklearn.ensemble import GradientBoostingRegressor

# إعداد النموذج
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# تدريب النموذج
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
import numpy as np

# حساب RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

