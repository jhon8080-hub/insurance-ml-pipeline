"""
Feature Importance CatBoost (src/CB_importance.py)
Результаты: metrics/CB_feature_importance.csv,
            models/CB_feature_importance.png,
            metrics/CB_importance_analysis.json
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

model   = CatBoostRegressor()
model.load_model('models/CB_model.cbm')
X_train = pd.read_csv('data/X_train.csv')

# Feature Importance (Таблица 4 отчёта)
fi = pd.DataFrame({
    'Признак':    X_train.columns,
    'Важность':   model.get_feature_importance(),
    'Важность, %': model.get_feature_importance(),
}).sort_values('Важность', ascending=False)

print("Таблица 4 – Feature Importance CatBoost:")
print(fi[['Признак','Важность, %']].round(2).to_string(index=False))

fi.to_csv('metrics/CB_feature_importance.csv', index=False)

# Рисунок 5 – График Feature Importance CatBoost
fi_plot = fi.sort_values('Важность')
fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
colors = ['#1A5276' if v >= fi['Важность'].median() else '#2980B9'
          for v in fi_plot['Важность']]
bars = ax.barh(fi_plot['Признак'], fi_plot['Важность'],
               color=colors, edgecolor='white', height=0.65)
for bar, val in zip(bars, fi_plot['Важность']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=9)
ax.set_xlabel('Важность (%)', fontsize=11)
ax.set_title('Рисунок 5 – Feature Importance CatBoost',
             fontsize=13, fontweight='bold', pad=12)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
ax.set_xlim(0, fi['Важность'].max() * 1.2)
plt.tight_layout()
plt.savefig('models/CB_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

# Сохранение анализа
analysis = {
    'top1_feature': fi.iloc[0]['Признак'],
    'top1_importance': round(fi.iloc[0]['Важность'], 2),
    'top2_feature': fi.iloc[1]['Признак'],
    'top2_importance': round(fi.iloc[1]['Важность'], 2),
    'top3_feature': fi.iloc[2]['Признак'],
    'top3_importance': round(fi.iloc[2]['Важность'], 2),
    'top3_cumulative': round(fi.iloc[:3]['Важность'].sum(), 2),
}
with open('metrics/CB_importance_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print(f"\nСохранено: metrics/CB_feature_importance.csv")
print(f"Сохранено: models/CB_feature_importance.png")
print(f"Сохранено: metrics/CB_importance_analysis.json")
