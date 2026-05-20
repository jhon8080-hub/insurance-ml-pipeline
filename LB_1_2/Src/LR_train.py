"""
Линейная регрессия (src/LR_train.py)
Датасет: insurance.csv — предсказание медицинских расходов
"""
import os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

X_train = pd.read_csv('data/X_train_scaled.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
mae  = mean_absolute_error(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
r2   = r2_score(y_train, y_pred)

print(f"Результаты обучения:")
print(f"  MAE:  ${mae:,.2f}  (средняя абсолютная ошибка)")
print(f"  RMSE: ${rmse:,.2f}  (среднеквадратичная ошибка)")
print(f"  R²:   {r2:.4f}")
print(f"\nСвободный член b₀ = ${model.intercept_:,.2f}")

coef_df = pd.DataFrame({'Признак': X_train.columns, 'Вес': model.coef_}
                       ).sort_values('Вес', ascending=False)
print(f"\nКоэффициенты модели:\n{coef_df.to_string(index=False)}")

with open('metrics/LR_metrics.json', 'w') as f:
    json.dump({'model':'LinearRegression','dataset':'train',
               'MAE':round(mae,2),'RMSE':round(rmse,2),'R2':round(r2,4),
               'intercept':round(model.intercept_,2)}, f, indent=2)

joblib.dump(model, 'models/LR_model.pkl')
print(f"\nМодель: models/LR_model.pkl\nМетрики: metrics/LR_metrics.json")
