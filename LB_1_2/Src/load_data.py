"""
Этап 1: Загрузка данных (src/load_data.py)
Датасет: insurance.csv — медицинские страховые расходы (США)
"""
import os, shutil, pandas as pd

os.makedirs('data', exist_ok=True)
shutil.copy2('data/insurance.csv', 'data/raw_data.csv')

df = pd.read_csv('data/raw_data.csv')
print(f"Загрузка успешна! Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"\nПервые строки:\n{df.head()}")
print(f"\nТипы данных:\n{df.dtypes}")
print(f"\nПропущенные значения: {df.isnull().sum().sum()}")
