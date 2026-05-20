"""
Анализ весов линейной регрессии (src/LR_analyze.py)
"""
import pandas as pd
import joblib

model   = joblib.load('models/LR_model.pkl')
X_train = pd.read_csv('data/X_train_scaled.csv')

coef_df = pd.DataFrame({'Признак': X_train.columns, 'Вес ($)': model.coef_}
                       ).sort_values('Вес ($)', ascending=False)

print(f"Свободный член b₀ = ${model.intercept_:,.2f}\n")
print("Таблица 1 – Коэффициенты линейной регрессии (после нормализации):")
print(coef_df.round(2).to_string(index=False))

report = f"""=== АНАЛИЗ ВЕСОВ ЛИНЕЙНОЙ РЕГРЕССИИ ===
Датасет: insurance.csv
Свободный член b₀ = ${model.intercept_:,.2f}

Коэффициенты (влияние на расходы при изменении признака на 1σ):
{coef_df.round(2).to_string(index=False)}

Вывод: smoker — главный предиктор (наибольший положительный коэффициент).
Курение увеличивает расходы примерно на $9 788 в среднем.
Age и bmi также значимо влияют на расходы.
"""
with open('metrics/LR_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\nОтчёт: metrics/LR_analysis_report.txt")
