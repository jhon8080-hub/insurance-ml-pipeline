"""
Реализация нейронной сети для регрессии (предсказание charges)
Архитектура: 9 → 128 → 64 → 32 → 1
Параметры из params.yaml: nn_epochs=500, nn_batch_size=32, nn_learning_rate=0.001
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Подавление предупреждений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def create_model(input_dim, learning_rate=0.001):
    """Создает архитектуру нейронной сети для регрессии"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,), name='hidden1'),
        layers.Dropout(0.3, name='dropout1'),
        layers.Dense(64, activation='relu', name='hidden2'),
        layers.Dropout(0.2, name='dropout2'),
        layers.Dense(32, activation='relu', name='hidden3'),
        layers.Dense(1, name='output')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def main():
    print("=" * 80)
    print("НЕЙРОННАЯ СЕТЬ - ОБУЧЕНИЕ (РЕГРЕССИЯ)")
    print("=" * 80)

    # Создаем папки
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('tb_logs', exist_ok=True)

    # Загружаем параметры из params.yaml
    import yaml
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    nn_epochs = params.get('nn_epochs', 500)
    nn_batch_size = params.get('nn_batch_size', 32)
    nn_learning_rate = params.get('nn_learning_rate', 0.001)

    print(f"\n📋 Параметры из params.yaml:")
    print(f"   epochs: {nn_epochs}")
    print(f"   batch_size: {nn_batch_size}")
    print(f"   learning_rate: {nn_learning_rate}")

    # Загружаем данные
    print("\n📥 Загружаем данные...")
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    X_val = pd.read_csv('data/X_val.csv')
    y_val = pd.read_csv('data/y_val.csv').squeeze()
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()

    # Нормализуем данные (важно для нейросети)
    print("\n🔄 Нормализуем данные...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Масштабируем целевую переменную
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

    # Сохраняем scaler'ы
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')

    print(f"   X_train: {X_train_scaled.shape}")
    print(f"   X_val:   {X_val_scaled.shape}")
    print(f"   X_test:  {X_test_scaled.shape}")

    # Создаем модель
    print("\n🧠 Создаем нейронную сеть...")
    model = create_model(X_train_scaled.shape[1], nn_learning_rate)
    model.summary()

    # Колбэки
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=f'tb_logs/nn_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=0.00001,
        verbose=1
    )

    # Обучаем модель
    print("\n⚡ Обучаем нейронную сеть...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=nn_epochs,
        batch_size=nn_batch_size,
        callbacks=[tensorboard_callback, early_stopping, reduce_lr],
        verbose=1
    )

    # Сохраняем модель
    model.save('models/nn_model.keras')
    print(f"\n✅ Модель сохранена в models/nn_model.keras")

    # Сохраняем историю обучения
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('metrics/nn_training_history.csv', index=False)

    # Рисуем кривые обучения
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Кривые обучения - Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Кривые обучения - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (нормализованная)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/nn_learning_curves.png', dpi=300)
    print("✅ Кривые обучения сохранены в plots/nn_learning_curves.png")

    # Оценка на train (в исходных единицах)
    y_train_pred_scaled = model.predict(X_train_scaled).ravel()
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    print("\n📊 МЕТРИКИ НА ОБУЧЕНИИ:")
    print(f"   MAE:  ${train_mae:,.0f}")
    print(f"   RMSE: ${train_rmse:,.0f}")
    print(f"   R²:   {train_r2:.4f}")

    # Сохраняем метрики
    metrics = {
        'model_type': 'NeuralNetwork',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': {
            'epochs': nn_epochs,
            'batch_size': nn_batch_size,
            'learning_rate': nn_learning_rate
        },
        'architecture': [
            {'layer': 'Dense128', 'activation': 'relu', 'dropout': 0.3},
            {'layer': 'Dense64', 'activation': 'relu', 'dropout': 0.2},
            {'layer': 'Dense32', 'activation': 'relu'},
            {'layer': 'Dense1', 'activation': 'linear'}
        ],
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'train_r2': float(train_r2),
        'n_features': X_train_scaled.shape[1],
        'n_samples': X_train_scaled.shape[0],
        'epochs_trained': len(history.history['loss'])
    }

    with open('metrics/nn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("   ✅ Метрики сохранены")

    print("\n" + "=" * 80)
    print("🎯 Обучение завершено! Запустите TensorBoard:")
    print("   tensorboard --logdir tb_logs/")
    print("=" * 80)


if __name__ == "__main__":
    main()