import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

# مثال على البيانات اللي غنخدمو بها
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=365),
    'Solar Production (kW)': np.random.randint(450, 600, size=365),
    'Wind Production (kW)': np.random.randint(250, 350, size=365),
    'Wind Speed (km/h)': np.random.uniform(10, 30, size=365),
    'Solar Irradiance (W/m²)': np.random.uniform(400, 800, size=365),
    'Temperature (°C)': np.random.uniform(15, 35, size=365),
    'Humidity (%)': np.random.uniform(30, 70, size=365),
    'Pressure (hPa)': np.random.uniform(950, 1050, size=365)
})

# تحضير البيانات
X = data[['Wind Speed (km/h)', 'Solar Irradiance (W/m²)', 'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)']]
y = data['Solar Production (kW)']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء XGBoost model
xg_model = xgb.XGBRegressor()

# إعداد المعاملات اللي بغينا نقلبو عليها
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0]
}

# استخدام GridSearchCV للبحث عن أفضل المعاملات
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=xg_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# أفضل المعاملات اللي حصلنا عليها
print("Best parameters:", grid_search.best_params_)

# التنبؤ على البيانات الجديدة
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# حساب الـ MSE و RMSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# إنشاء DataFrame جديد باش نقارنو التنبؤات بالقيم الحقيقية
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# إنشاء الرسم البياني
plt.figure(figsize=(10,6))
plt.plot(results.index, results['Actual'], label='Actual Solar Production', color='blue')
plt.plot(results.index, results['Predicted'], label='Predicted Solar Production', color='red', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Solar Production (kW)')
plt.title('Comparison of Actual vs Predicted Solar Production')
plt.legend()
plt.grid(True)
plt.show()
