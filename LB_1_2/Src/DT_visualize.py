"""
Визуализация дерева решений (src/DT_visualize.py)
Назначение: создание графического представления первых уровней дерева
            для понимания логики принятия решений.
Результат: models/DT_tree_structure.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

os.makedirs('models', exist_ok=True)

model   = joblib.load('models/DT_model.pkl')
X_train = pd.read_csv('data/X_train.csv')

# Рисунок 4 – Визуализация первых 3 уровней дерева решений
fig, ax = plt.subplots(figsize=(22, 8), facecolor='white')
fig.suptitle('Рисунок 4 – Визуализация первых узлов дерева решений',
             fontsize=13, fontweight='bold')

plot_tree(
    model,
    max_depth=3,
    feature_names=X_train.columns.tolist(),
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    impurity=True,
    precision=2,
)

plt.tight_layout()
plt.savefig('models/DT_tree_structure.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

print("Рисунок сохранён: models/DT_tree_structure.png")
print(f"\nКорневой узел: разбиение по признаку "
      f"'{X_train.columns[model.tree_.feature[0]]}'")
print("\nЛевая ветка  (≤ порога): низкая взрослая смертность → высокая продолж. жизни")
print("Правая ветка (> порога): высокая взрослая смертность → низкая продолж. жизни")
