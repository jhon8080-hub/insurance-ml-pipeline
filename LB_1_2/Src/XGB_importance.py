"""
Feature Importance XGBoost (src/XGB_importance.py)
Результаты: metrics/XGB_feature_importance.csv,
            models/XGB_feature_importance.png,
            metrics/XGB_importance_analysis.json
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

model   = XGBRegressor()
model.load_model('models/XGB_model.json')
X_train = pd.read_csv('data/X_train.csv')

# Feature Importance (Таблица 6 отчёта)
fi = pd.DataFrame({
    'Признак':      X_train.columns,
    'Важность':     model.feature_importances_,
    'Важность, %':  model.feature_importances_ * 100,
}).sort_values('Важность', ascending=False)

print("Таблица 6 – Feature Importance XGBoost:")
print(fi[['Признак','Важность','Важность, %']].round(4).to_string(index=False))

fi.to_csv('metrics/XGB_feature_importance.csv', index=False)

# Рисунок 6 – График Feature Importance XGBoost
fi_plot = fi.sort_values('Важность, %')
fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
colors = ['#7B241C' if v >= fi['Важность, %'].median() else '#C0392B'
          for v in fi_plot['Важность, %']]
bars = ax.barh(fi_plot['Признак'], fi_plot['Важность, %'],
               color=colors, edgecolor='white', height=0.65)
for bar, val in zip(bars, fi_plot['Важность, %']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=9)
ax.set_xlabel('Важность (%)', fontsize=11)
ax.set_title('Рисунок 6 – Feature Importance XGBoost',
             fontsize=13, fontweight='bold', pad=12)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
ax.set_xlim(0, fi['Важность, %'].max() * 1.2)
plt.tight_layout()
plt.savefig('models/XGB_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

# Сохранение анализа
analysis = {
    'top1_feature':    fi.iloc[0]['Признак'],
    'top1_importance': round(fi.iloc[0]['Важность, %'], 2),
    'top2_feature':    fi.iloc[1]['Признак'],
    'top2_importance': round(fi.iloc[1]['Важность, %'], 2),
    'top3_feature':    fi.iloc[2]['Признак'],
    'top3_importance': round(fi.iloc[2]['Важность, %'], 2),
    'top3_cumulative': round(fi.iloc[:3]['Важность, %'].sum(), 2),
    'note': 'XGBoost концентрируется на 3 ключевых признаках (>94% суммарной важности)',
}
with open('metrics/XGB_importance_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print(f"\nСохранено: metrics/XGB_feature_importance.csv")
print(f"Сохранено: models/XGB_feature_importance.png")
print(f"Сохранено: metrics/XGB_importance_analysis.json")
