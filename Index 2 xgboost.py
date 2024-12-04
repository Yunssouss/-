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


# تغيير أسماء الأعمدة
data.columns = ['Date', 'Solar Production (kW)', 'Wind Production (kW)', 'Wind Speed (km/h)', 'Solar Radiation (W/m²)', 'Temperature (°C)']

# الآن نقدر نستخدم الأسماء الجديدة
X = data[['Wind Speed (km/h)', 'Solar Radiation (W/m²)', 'Temperature (°C)']]  # الخصائص
y = data['Solar Production (kW)']  # القيمة المستهدفة

# تقسيم المعطيات
from sklearn.model_selection import train_test_split
# تقسيم المعطيات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج باستخدام XGBoost
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred = model.predict(X_test)

# حساب MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# عرض نتائج التنبؤات
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())





import xgboost as xgb

# تحويل المعطيات إلى DMatrix (الهيكلية الخاصة بXGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# إعداد المعاملات
params = {
    'max_depth': 6,         # العمق ديال الشجرة
    'eta': 0.1,             # سرعة التعلم
    'objective': 'reg:squarederror',  # الهدف هو التنبؤ بالقيم العددية
    'eval_metric': 'rmse'   # مقياس الأداء
}

# تدريب النموذج
bst = xgb.train(params, dtrain, num_boost_round=100)

# التنبؤ
y_pred = bst.predict(dtest)

from sklearn.metrics import mean_squared_error
import numpy as np

import pandas as pd

# تحويل القيم الحقيقية والتنبؤات إلى DataFrame باش نعرضهم بشكل منظم
results = pd.DataFrame({
    'التاريخ': data.index[X_test.index],  # نعرض التواريخ اللي تنبأنا فيها
    'القيمة الحقيقية لإنتاج الطاقة': y_test,
    'التنبؤ بإنتاج الطاقة': y_pred
})

# عرض أول 10 نتائج
print(results.head(10))


# حساب RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")


import matplotlib.pyplot as plt

# إنشاء الرسم البياني
plt.figure(figsize=(10,6))
plt.plot(results['التاريخ'], results['القيمة الحقيقية لإنتاج الطاقة'], label='القيمة الحقيقية', color='blue')
plt.plot(results['التاريخ'], results['التنبؤ بإنتاج الطاقة'], label='التنبؤ بإنتاج الطاقة', color='red', linestyle='--')

# تخصيص الرسم
plt.xlabel('التاريخ')
plt.ylabel('إنتاج الطاقة (kW)')
plt.title('Comparison of Actual vs Predicted Solar Productionة')
plt.legend()
plt.grid(True)

# عرض الرسم
plt.show()




