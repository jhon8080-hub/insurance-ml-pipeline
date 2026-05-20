"""
Дерево решений (src/DT_train.py)
Параметры: max_depth=6, min_samples_leaf=10, random_state=42
"""
import os, json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
mae  = mean_absolute_error(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
r2   = r2_score(y_train, y_pred)

print(f"Результаты обучения:")
print(f"  MAE:  ${mae:,.2f}")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  R²:   {r2:.4f}")
print(f"\nКорневой узел: {X_train.columns[model.tree_.feature[0]]}")

with open('metrics/DT_metrics.json', 'w') as f:
    json.dump({'model':'DecisionTreeRegressor','dataset':'train',
               'max_depth':6,'min_samples_leaf':10,
               'MAE':round(mae,2),'RMSE':round(rmse,2),'R2':round(r2,4)}, f, indent=2)

joblib.dump(model, 'models/DT_model.pkl')
print(f"\nМодель: models/DT_model.pkl")
