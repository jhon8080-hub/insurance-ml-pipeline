"""
Анализ XGBoost (src/XGB_analyze.py)
Результат: metrics/XGB_analysis_report.txt
"""

import json
import pandas as pd

with open('metrics/XGB_metrics.json') as f:
    train_m = json.load(f)
with open('metrics/XGB_test_metrics.json') as f:
    test_m  = json.load(f)

fi = pd.read_csv('metrics/XGB_feature_importance.csv')

report = [
    "=== АНАЛИЗ XGBOOST ===\n",
    "Параметры: n_estimators=300, learning_rate=0.05, max_depth=6,",
    "           subsample=0.8, colsample_bytree=0.8\n",
    f"Метрики обучения: MAE={train_m['MAE']}, RMSE={train_m['RMSE']}, R²={train_m['R2']}",
    f"Метрики теста:    MAE={test_m['MAE']}, RMSE={test_m['RMSE']}, R²={test_m['R2']}",
    f"Переобучение:     {test_m['overfit']} (~{test_m['overfit']*100:.1f}%)\n",
    "\nТоп Feature Importance:",
    fi[['Признак','Важность, %']].round(2).head(5).to_string(index=False),
    "\n\nВывод: XGBoost — лучшая модель эксперимента.",
    "HIV/AIDS доминирует (77% важности) — XGBoost сосредоточился на",
    "наиболее дискриминативном признаке благодаря регуляризации L1/L2.",
]
with open('metrics/XGB_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print('\n'.join(report))
print(f"\nОтчёт сохранён: metrics/XGB_analysis_report.txt")
