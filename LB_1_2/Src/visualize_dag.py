"""
Визуализация вычислительного графа DVC (src/visualize_dag.py)
Создает цветной граф пайплайна с помощью matplotlib
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import os

def create_dag_visualization():
    """Создает визуальное представление DAG графа"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Определяем позиции узлов
    nodes = {
        # Подготовка данных
        'load_data': (6, 12.5),
        'preprocess': (6, 11),
        'feature_engineering': (6, 9.5),
        'split_data': (6, 8),
        
        # Модели (обучение)
        'LR_train': (1, 6.5),
        'DT_train': (3, 6.5),
        'CB_train': (5, 6.5),
        'XGB_train': (7, 6.5),
        'NN_train': (9, 6.5),
        
        # Валидация
        'LR_val': (1, 5),
        'DT_val': (3, 5),
        'CB_val': (5, 5),
        'XGB_val': (7, 5),
        'NN_val': (9, 5),
        
        # Тестирование
        'LR_test': (1, 3.5),
        'DT_test': (3, 3.5),
        'CB_test': (5, 3.5),
        'XGB_test': (7, 3.5),
        'NN_test': (9, 3.5),
        
        # Анализ
        'LR_analyze': (1, 2),
        'DT_analyze': (3, 2),
        'CB_analyze': (5, 2),
        'XGB_analyze': (7, 2),
        'NN_analyze': (9, 2),
        
        # Сравнение моделей
        'compare_models': (5, 0.5)
    }
    
    # Цвета для разных типов этапов
    colors = {
        'data': '#3498db',      # синий - подготовка данных
        'linear': '#2ecc71',    # зеленый - линейная регрессия
        'tree': '#e74c3c',      # красный - дерево решений
        'catboost': '#9b59b6',  # фиолетовый - CatBoost
        'xgboost': '#f39c12',   # оранжевый - XGBoost
        'neural': '#1abc9c',    # бирюзовый - нейронная сеть
        'compare': '#34495e'    # темно-синий - сравнение
    }
    
    # Определяем цвет для каждого узла
    node_colors = {}
    for node in nodes:
        if node in ['load_data', 'preprocess', 'feature_engineering', 'split_data']:
            node_colors[node] = colors['data']
        elif 'LR' in node:
            node_colors[node] = colors['linear']
        elif 'DT' in node:
            node_colors[node] = colors['tree']
        elif 'CB' in node:
            node_colors[node] = colors['catboost']
        elif 'XGB' in node:
            node_colors[node] = colors['xgboost']
        elif 'NN' in node:
            node_colors[node] = colors['neural']
        else:
            node_colors[node] = colors['compare']
    
    # Рисуем узлы
    for node, pos in nodes.items():
        rect = FancyBboxPatch(
            (pos[0] - 1.2, pos[1] - 0.4), 2.4, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=node_colors[node],
            alpha=0.85,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Название узла (с переносом строки)
        if node == 'feature_engineering':
            label = 'feature\nengineering'
        elif node == 'compare_models':
            label = 'compare\nmodels'
        else:
            label = node.replace('_', '\n')
        
        ax.text(pos[0], pos[1], label, 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Определяем связи (стрелки)
    connections = [
        # Подготовка данных (вертикальные связи)
        ('load_data', 'preprocess'),
        ('preprocess', 'feature_engineering'),
        ('feature_engineering', 'split_data'),
        
        # От split_data к обучению моделей
        ('split_data', 'LR_train'),
        ('split_data', 'DT_train'),
        ('split_data', 'CB_train'),
        ('split_data', 'XGB_train'),
        ('split_data', 'NN_train'),
        
        # Цепочка для Linear Regression
        ('LR_train', 'LR_val'),
        ('LR_val', 'LR_test'),
        ('LR_test', 'LR_analyze'),
        
        # Цепочка для Decision Tree
        ('DT_train', 'DT_val'),
        ('DT_val', 'DT_test'),
        ('DT_test', 'DT_analyze'),
        
        # Цепочка для CatBoost
        ('CB_train', 'CB_val'),
        ('CB_val', 'CB_test'),
        ('CB_test', 'CB_analyze'),
        
        # Цепочка для XGBoost
        ('XGB_train', 'XGB_val'),
        ('XGB_val', 'XGB_test'),
        ('XGB_test', 'XGB_analyze'),
        
        # Цепочка для Neural Network
        ('NN_train', 'NN_val'),
        ('NN_val', 'NN_test'),
        ('NN_test', 'NN_analyze'),
        
        # Все анализы → сравнение моделей
        ('LR_analyze', 'compare_models'),
        ('DT_analyze', 'compare_models'),
        ('CB_analyze', 'compare_models'),
        ('XGB_analyze', 'compare_models'),
        ('NN_analyze', 'compare_models'),
    ]
    
    # Рисуем стрелки
    for start, end in connections:
        start_pos = nodes[start]
        end_pos = nodes[end]
        
        # Определяем координаты стрелок
        if start_pos[1] > end_pos[1] + 0.5:  # Вертикальная связь вниз
            arrow_start = (start_pos[0], start_pos[1] - 0.4)
            arrow_end = (end_pos[0], end_pos[1] + 0.4)
        elif start_pos[0] < end_pos[0]:  # Горизонтальная связь вправо
            arrow_start = (start_pos[0] + 1.2, start_pos[1])
            arrow_end = (end_pos[0] - 1.2, end_pos[1])
        elif start_pos[0] > end_pos[0]:  # Горизонтальная связь влево
            arrow_start = (start_pos[0] - 1.2, start_pos[1])
            arrow_end = (end_pos[0] + 1.2, end_pos[1])
        else:  # Вертикальная связь
            arrow_start = (start_pos[0], start_pos[1] - 0.4)
            arrow_end = (end_pos[0], end_pos[1] + 0.4)
        
        ax.annotate('', xy=arrow_end, xytext=arrow_start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.7))
    
    # Легенда
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], alpha=0.85, label='Подготовка данных'),
        mpatches.Patch(facecolor=colors['linear'], alpha=0.85, label='Linear Regression'),
        mpatches.Patch(facecolor=colors['tree'], alpha=0.85, label='Decision Tree'),
        mpatches.Patch(facecolor=colors['catboost'], alpha=0.85, label='CatBoost'),
        mpatches.Patch(facecolor=colors['xgboost'], alpha=0.85, label='XGBoost'),
        mpatches.Patch(facecolor=colors['neural'], alpha=0.85, label='Neural Network'),
        mpatches.Patch(facecolor=colors['compare'], alpha=0.85, label='Сравнение моделей')
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', 
              bbox_to_anchor=(0.02, 0.02), fontsize=9, framealpha=0.9)
    
    # Заголовок
    plt.title('Вычислительный граф DVC пайплайна', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Сохраняем изображение
    output_path = 'plots/dvc_graph_custom.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Граф DVC сохранён: {output_path}")
    print(f"   Размер: 16x12 дюймов, 300 dpi")


def main():
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ВЫЧИСЛИТЕЛЬНОГО ГРАФА DVC")
    print("=" * 60)
    
    create_dag_visualization()
    
    print("\n" + "=" * 60)
    print("✅ Готово! Граф можно вставить в отчёт.")
    print("   Файл: plots/dvc_graph_custom.png")
    print("=" * 60)


if __name__ == "__main__":
    main()