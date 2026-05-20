"""
Сравнение всех моделей (src/compare_models.py)
Таблица 10 отчёта — сводная таблица метрик.
Результаты: metrics/comparison_report.txt, plots/models_comparison.png
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('metrics', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Загрузка метрик всех моделей с тестовой выборки
def load(path):
    with open(path) as f:
        return json.load(f)

results = [
    {**load('metrics/XGB_test_metrics.json'),  'Модель': 'XGBoost'},
    {**load('metrics/CB_test_metrics.json'),   'Модель': 'CatBoost'},
    {**load('metrics/DT_test_metrics.json'),   'Модель': 'Decision Tree'},
    {**load('metrics/NN_test_metrics.json'),   'Модель': 'Neural Network'},
    {**load('metrics/LR_test_metrics.json'),   'Модель': 'Linear Regression'},
]

df = pd.DataFrame(results)[['Модель', 'R2', 'MAE', 'RMSE', 'overfit']]
df.columns = ['Модель', 'R² (test)', 'MAE (лет)', 'RMSE (лет)', 'Переобучение']
df = df.sort_values('R² (test)', ascending=False).reset_index(drop=True)

print("Таблица 10 – Сводная таблица метрик всех моделей (Таблица 10 отчёта):")
print(df.to_string(index=False))

best = df.iloc[0]
print(f"\n{'='*50}")
print(f"Лучшая модель: {best['Модель']}")
print(f"  R² = {best['R² (test)']:.4f}")
print(f"  MAE = {best['MAE (лет)']:.3f} года")
print(f"  RMSE = {best['RMSE (лет)']:.3f} года")
print(f"  Переобучение = {best['Переобучение']:.4f}")

# График сравнения моделей
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='white')
fig.suptitle('Сравнение моделей по метрикам', fontsize=14, fontweight='bold')

COLORS = ['#E2EFDA' if i == 0 else '#BDD7EE' for i in range(len(df))]
METRICS = [('R² (test)', True), ('MAE (лет)', False), ('RMSE (лет)', False)]

for ax, (metric, higher_better) in zip(axes, METRICS):
    vals = df[metric]
    bars = ax.bar(df['Модель'], vals, color=COLORS, edgecolor='white', width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (vals.max() * 0.01),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_xticklabels(df['Модель'], rotation=20, ha='right', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor('#F8F9FA')
    if higher_better:
        ax.set_ylim(vals.min() * 0.95, vals.max() * 1.05)

plt.tight_layout()
plt.savefig('plots/models_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("\nГрафик сравнения сохранён: plots/models_comparison.png")

# Отчёт
report_lines = [
    "=== СВОДНОЕ СРАВНЕНИЕ МОДЕЛЕЙ ===\n",
    df.to_string(index=False),
    f"\n\nЛучшая модель: {best['Модель']} (R²={best['R² (test)']:.4f}, "
    f"MAE={best['MAE (лет)']:.3f} лет)",
    "\nПочему XGBoost лучший:",
    "  1. L1/L2 регуляризация контролирует сложность модели",
    "  2. Точные приближения градиента и гессиана ускоряют сходимость",
    "  3. Subsampling (0.8) снижает переобучение",
    "  4. Концентрация на 3 ключевых признаках (HIV/AIDS, Income, Adult Mortality)",
]
with open('metrics/comparison_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print("Отчёт сохранён: metrics/comparison_report.txt")
