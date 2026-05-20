"""
Реализация XGBoost (src/XGB_train.py)
Параметры обучения (Таблица 5 отчёта):
  n_estimators=300, learning_rate=0.05, max_depth=6,
  subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30
"""

import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
X_val   = pd.read_csv('data/X_val.csv')
y_val   = pd.read_csv('data/y_val.csv').squeeze()

print(f"Обучающая выборка: {X_train.shape}")

# Обучение с параметрами из Таблицы 5
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,
    random_state=42,
    verbosity=1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# Метрики на обучающей выборке
y_pred_train = model.predict(X_train)
mae  = mean_absolute_error(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2   = r2_score(y_train, y_pred_train)

print(f"\nРезультаты обучения:")
print(f"  MAE:  {mae:.3f} года")
print(f"  RMSE: {rmse:.3f} года")
print(f"  R²:   {r2:.4f}")

metrics = {
    'model': 'XGBRegressor',
    'dataset': 'train',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'MAE':  round(mae, 4),
    'RMSE': round(rmse, 4),
    'R2':   round(r2, 4),
}
with open('metrics/XGB_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Сохранение модели в формате .json
model.save_model('models/XGB_model.json')

print(f"\nМодель сохранена: models/XGB_model.json")
print(f"Метрики сохранены: metrics/XGB_metrics.json")
