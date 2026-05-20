"""
Валидация линейной регрессии (src/LR_val.py)
"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

model  = joblib.load('models/LR_model.pkl')
X_val  = pd.read_csv('data/X_val_scaled.csv')
y_val  = pd.read_csv('data/y_val.csv').squeeze()

y_pred = model.predict(X_val)
mae  = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2   = r2_score(y_val, y_pred)

print(f"Результаты валидации:")
print(f"  MAE:  ${mae:,.2f}")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  R²:   {r2:.4f}")

with open('metrics/LR_val_metrics.json', 'w') as f:
    json.dump({'model':'LinearRegression','dataset':'val',
               'MAE':round(mae,2),'RMSE':round(rmse,2),'R2':round(r2,4)}, f, indent=2)
print("Метрики: metrics/LR_val_metrics.json")
