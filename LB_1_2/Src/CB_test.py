"""
Тестирование CatBoost (src/CB_test.py)
"""

import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model  = CatBoostRegressor()
model.load_model('models/CB_model.cbm')

X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

y_pred_test = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

with open('metrics/CB_metrics.json') as f:
    train_r2 = json.load(f)['R2']
overfit = round(train_r2 - r2, 4)

print(f"Результаты теста:")
print(f"  MAE:  {mae:.3f} года")
print(f"  RMSE: {rmse:.3f} года")
print(f"  R²:   {r2:.4f}")
print(f"  Переобучение: {overfit:.4f} (~{overfit*100:.1f}%)")

metrics = {
    'model': 'CatBoostRegressor',
    'dataset': 'test',
    'MAE':  round(mae, 4),
    'RMSE': round(rmse, 4),
    'R2':   round(r2, 4),
    'overfit': overfit,
}
with open('metrics/CB_test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"\nМетрики сохранены: metrics/CB_test_metrics.json")
