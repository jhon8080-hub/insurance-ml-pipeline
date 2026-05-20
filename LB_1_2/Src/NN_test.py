"""
Тестирование нейронной сети (src/NN_test.py)
Загружает Keras-модель, сохранённую в NN_train.py.
"""
import json
import numpy as np
import pandas as pd
import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model  = tf.keras.models.load_model('models/NN_model.keras')
X_test = pd.read_csv('data/X_test_scaled.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

y_pred = model.predict(X_test.values, verbose=0).flatten()
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

with open('metrics/NN_metrics.json') as f:
    train_r2 = json.load(f)['R2']
overfit = round(train_r2 - r2, 4)

print(f"Результаты теста:")
print(f"  MAE:  ${mae:,.2f}")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  R²:   {r2:.4f}")
print(f"  Переобучение: {overfit:.4f} (~{overfit*100:.1f}%)")

with open('metrics/NN_test_metrics.json', 'w') as f:
    json.dump({'model':'Keras NN','dataset':'test',
               'MAE':round(mae,2),'RMSE':round(rmse,2),
               'R2':round(r2,4),'overfit':overfit}, f, indent=2)
print("Метрики: metrics/NN_test_metrics.json")
