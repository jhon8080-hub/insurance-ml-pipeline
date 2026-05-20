"""
Анализ важности признаков дерева решений (src/DT_analyze.py)
Результаты: metrics/DT_analysis_report.txt, metrics/DT_feature_importance.csv
"""

import pandas as pd
import joblib

model   = joblib.load('models/DT_model.pkl')
X_train = pd.read_csv('data/X_train.csv')

# Feature Importance
fi = pd.DataFrame({
    'Признак':    X_train.columns,
    'Важность':   model.feature_importances_,
    'Важность, %': model.feature_importances_ * 100,
}).sort_values('Важность', ascending=False)

print("Таблица 2 – Важность признаков (Feature Importance) дерева решений:")
print(fi[['Признак','Важность, %']].round(2).to_string(index=False))

fi.to_csv('metrics/DT_feature_importance.csv', index=False)

report = [
    "=== АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ — ДЕРЕВО РЕШЕНИЙ ===\n",
    fi[['Признак','Важность, %']].round(2).to_string(index=False),
    "\n\nВывод:",
    f"  Доминирующий признак: {fi.iloc[0]['Признак']} ({fi.iloc[0]['Важность, %']:.2f}%)",
    f"  Второй по важности:   {fi.iloc[1]['Признак']} ({fi.iloc[1]['Важность, %']:.2f}%)",
    "\nЭто соответствует корреляционному анализу EDA.",
]
with open('metrics/DT_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"\nСохранено: metrics/DT_feature_importance.csv")
print(f"Сохранено: metrics/DT_analysis_report.txt")
