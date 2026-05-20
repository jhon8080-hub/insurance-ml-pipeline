"""
Тестирование линейной регрессии (src/LR_test.py)
"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

model  = joblib.load('models/LR_model.pkl')
X_test = pd.read_csv('data/X_test_scaled.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

with open('metrics/LR_metrics.json') as f:
    train_r2 = json.load(f)['R2']
overfit = round(train_r2 - r2, 4)

print(f"Результаты теста:")
print(f"  MAE:  ${mae:,.2f}")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  R²:   {r2:.4f}")
print(f"  Переобучение: {overfit:.4f}")

with open('metrics/LR_test_metrics.json', 'w') as f:
    json.dump({'model':'LinearRegression','dataset':'test',
               'MAE':round(mae,2),'RMSE':round(rmse,2),'R2':round(r2,4),
               'overfit':overfit}, f, indent=2)
print("Метрики: metrics/LR_test_metrics.json")
