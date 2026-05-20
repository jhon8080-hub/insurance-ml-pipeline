"""
Этап 2: Предобработка данных (src/preprocess.py)
Кодирование категориальных признаков: sex, smoker → 0/1, region → one-hot
"""
import pandas as pd

df = pd.read_csv('data/raw_data.csv')
print(f"Исходный размер: {df.shape}")
print(f"Пропуски: {df.isnull().sum().sum()}")

# sex: male=1, female=0
df['sex'] = (df['sex'] == 'male').astype(int)
print(f"\nsex: male=1, female=0")

# smoker: yes=1, no=0
df['smoker'] = (df['smoker'] == 'yes').astype(int)
print(f"smoker: yes=1, no=0")

# region: one-hot encoding
df = pd.get_dummies(df, columns=['region'])
print(f"region: one-hot (4 дамми-переменные)")

# Конвертация булевых в int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"\nРазмер после обработки: {df.shape}")
print(f"Признаки: {df.columns.tolist()}")

df.to_csv('data/processed_data.csv', index=False)
print(f"\nСохранено: data/processed_data.csv")
