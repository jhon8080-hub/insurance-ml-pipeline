"""
Этап 4: Разделение данных (src/split_data.py)
Пропорция: 60% train / 20% val / 20% test
Нормализация StandardScaler для LR и NN.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

TARGET = 'charges'
RANDOM_STATE = 42
os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/featured_data.csv')
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

print(f"Обучающая выборка:    {X_train.shape[0]} записей")
print(f"Валидационная выборка: {X_val.shape[0]} записей")
print(f"Тестовая выборка:      {X_test.shape[0]} записей")

# StandardScaler для LR и NN
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_val_sc   = pd.DataFrame(scaler.transform(X_val),       columns=X.columns)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

X_train.to_csv('data/X_train.csv', index=False)
X_val.to_csv(  'data/X_val.csv',   index=False)
X_test.to_csv( 'data/X_test.csv',  index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_val.to_csv(  'data/y_val.csv',   index=False)
y_test.to_csv( 'data/y_test.csv',  index=False)

X_train_sc.to_csv('data/X_train_scaled.csv', index=False)
X_val_sc.to_csv(  'data/X_val_scaled.csv',   index=False)
X_test_sc.to_csv( 'data/X_test_scaled.csv',  index=False)

joblib.dump(scaler, 'models/scaler_X.pkl')
print(f"\nСохранены файлы в data/ и models/scaler_X.pkl")
