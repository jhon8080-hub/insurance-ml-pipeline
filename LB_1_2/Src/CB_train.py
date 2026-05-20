"""
Реализация CatBoost (src/CB_train.py)
Параметры обучения (Таблица 3 отчёта):
  iterations=300, learning_rate=0.05, depth=6,
  early_stopping_rounds=50, use_best_model=True
"""

import os
import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Загрузка данных (без нормализации — CatBoost не требует)
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
X_val   = pd.read_csv('data/X_val.csv')
y_val   = pd.read_csv('data/y_val.csv').squeeze()

print(f"Обучающая выборка: {X_train.shape}")

# Обучение с параметрами из Таблицы 3
model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=50,
    use_best_model=True,
    random_seed=42,
    verbose=50,
)
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
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

print(f"\nСравнение с предыдущими моделями:")
print(f"  Линейная регрессия: R² ≈ 0.7977")
print(f"  Дерево решений:     R² ≈ 0.9200")
print(f"  CatBoost:           R² = {r2:.4f}")

metrics = {
    'model': 'CatBoostRegressor',
    'dataset': 'train',
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 6,
    'MAE':  round(mae, 4),
    'RMSE': round(rmse, 4),
    'R2':   round(r2, 4),
}
with open('metrics/CB_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Сохранение модели в формате .cbm
model.save_model('models/CB_model.cbm')

print(f"\nМодель сохранена: models/CB_model.cbm")
print(f"Метрики сохранены: metrics/CB_metrics.json")
