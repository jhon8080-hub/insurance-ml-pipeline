"""
Полный анализ нейронной сети (src/NN_analyze.py)
Результат: metrics/NN_full_report.txt
"""
import json
import pandas as pd

with open('metrics/NN_metrics.json') as f:
    train_m = json.load(f)
with open('metrics/NN_test_metrics.json') as f:
    test_m  = json.load(f)
with open('metrics/NN_weights_analysis.json') as f:
    weights = json.load(f)

history = pd.read_csv('metrics/NN_training_history.csv')

report = [
    "=== ПОЛНЫЙ АНАЛИЗ НЕЙРОННОЙ СЕТИ (Keras) ===\n",
    f"Архитектура: вход(9) → 128 (ReLU) → 64 (ReLU) → 32 (ReLU) → выход(1)",
    f"Оптимизатор: Adam, lr={train_m['learning_rate']}, batch_size={train_m['batch_size']}",
    f"Запланировано эпох: {train_m['epochs_planned']}, "
    f"обучено: {train_m['epochs_actual']} (EarlyStopping patience=20)",
    f"TensorBoard логи: {train_m.get('tensorboard_logdir','tb_logs/')}  "
    f"(запуск: tensorboard --logdir=tb_logs)",
    f"\nМетрики обучения: MAE=${train_m['MAE']:,}, RMSE=${train_m['RMSE']:,}, R²={train_m['R2']}",
    f"Метрики теста:    MAE=${test_m['MAE']:,}, RMSE=${test_m['RMSE']:,}, R²={test_m['R2']}",
    f"Переобучение:     {test_m['overfit']} (~{test_m['overfit']*100:.1f}%)\n",
    "\nАнализ весов по слоям:",
]
for i in range(1, 5):
    layer = weights.get(f'layer_{i}')
    if layer:
        report.append(
            f"  Слой {i} ({layer.get('name','')}, {layer['n_weights']} весов): "
            f"μ={layer['mean']}, σ={layer['std']}, "
            f"+:{layer['pct_positive']}% -:{layer['pct_negative']}%"
        )

report.append("\nВыводы по распределению весов:")
for c in weights.get('conclusions', []):
    report.append(f"  • {c}")

report += [
    "\n\nДостоинства Keras NN:",
    "  1. Гибкая архитектура — легко менять слои и активации",
    "  2. TensorBoard интеграция — визуализация обучения в реальном времени",
    "  3. Нелинейные зависимости улавливаются лучше, чем линейной регрессией",
    "  4. Хорошая точность (R² > 0.81) при умеренном переобучении",
    "\nНедостатки:",
    "  1. Требует нормализации данных (StandardScaler)",
    "  2. На малом датасете (746 записей) уступает CatBoost",
    "  3. Много гиперпараметров для настройки",
]

text = '\n'.join(report)
print(text)
with open('metrics/NN_full_report.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f"\nОтчёт: metrics/NN_full_report.txt")
