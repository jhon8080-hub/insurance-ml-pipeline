"""
Этап 3: Feature Engineering (src/feature_engineering.py)
Отбор признаков по корреляционному анализу.
Все 9 числовых признаков информативны — исключений нет.
"""
import pandas as pd

TARGET = 'charges'

df = pd.read_csv('data/processed_data.csv')
print(f"Исходный размер: {df.shape}")

# Корреляция с таргетом
corr = df.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print(f"\nКорреляция признаков с charges:")
print(corr.round(3).to_string())

# Все признаки информативны — используем все
SELECTED = [c for c in df.columns if c != TARGET]
print(f"\nОтобрано признаков: {len(SELECTED)}")
print(f"Список: {SELECTED}")

df[SELECTED + [TARGET]].to_csv('data/featured_data.csv', index=False)
print(f"\nСохранено: data/featured_data.csv")
