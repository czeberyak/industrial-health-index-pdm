import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptive_monitor import AdaptiveHealthMonitor  # Импортируйте ваш класс

# --- 1. Конфигурация ---
CONFIG = {
    'baseline_size': 200,  # Размер базовой линии (подберите под длину теста)
    'n_bands': 50,
    'k_sigma': 3.0,
    'fs': 20000,           # Частота дискретизации IMS
    'n_points': 20480      # Ожидаемая длина сигнала
}

DATA_DIR = '../data/4th_test'
BEARING_COL = 3 # Анализируем Bearing 1 (колонка 0)

# --- 2. Вспомогательные функции ---
def get_file_list(directory):
    """Получение списка файлов с фильтрацией мусора."""
    all_files = glob.glob(os.path.join(directory, '*'))
    excluded = ('.npy', '.png', '.zip', '.DS_Store', '.csv')
    valid = [f for f in all_files 
             if os.path.isfile(f) 
             and not f.endswith(excluded) 
             and ':Zone.Identifier' not in f 
             and not os.path.basename(f).startswith('.')]
    return sorted(valid)

def load_signal(filepath, col_idx=0):
    """Загрузка сигнала с проверкой целостности."""
    try:
        # 1st_test может иметь разделители табуляцией или пробелами
        df = pd.read_csv(filepath, sep=r'\s+', header=None, 
                         encoding='utf-8', on_bad_lines='skip')
        
        if col_idx >= df.shape[1]:
            return None
            
        signal = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
        signal = signal[np.isfinite(signal)]
        
        # Обрезка или проверка длины
        if len(signal) < CONFIG['n_points']:
            return None
        return signal[:CONFIG['n_points']]
    except Exception:
        return None

# --- 3. Запуск пайплайна ---
print(f"🚀 Запуск на {DATA_DIR} | Bearing Col: {BEARING_COL}")
monitor = AdaptiveHealthMonitor(CONFIG)
files = get_file_list(DATA_DIR)
print(f"📂 Найдено файлов: {len(files)}")

hi_scores = []
timestamps = []
calibration_done = False

for i, fpath in enumerate(files):
    signal = load_signal(fpath, BEARING_COL)
    if signal is None:
        continue
        
    res = monitor.process(signal, timestamp=os.path.basename(fpath))
    
    # Логирование калибровки
    if res['status'] == 'CALIBRATION_COMPLETE':
        print(f"✅ Калибровка завершена на файле {i}. Порог: {res['threshold']:.4f}")
        calibration_done = True
        
    if res['hi_score'] is not None:
        hi_scores.append(res['hi_score'])
        timestamps.append(i)
        
    # Опционально: логирование первой аномалии
    if calibration_done and res['is_anomaly'] and len(hi_scores) == sum(1 for x in hi_scores if x > monitor.threshold):
         print(f"⚠️ Первая детекция аномалии на файле {i} (HI: {res['hi_score']:.2f})")

# --- 4. Визуализация и Валидация ---
if hi_scores:
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, hi_scores, label='Health Index', color='#8c00ff')
    plt.axhline(y=monitor.threshold, color='r', linestyle='--', label=f'Threshold ({monitor.threshold:.2f})')
    
    # Отметка конца калибровки
    plt.axvline(x=CONFIG['baseline_size'], color='g', linestyle=':', alpha=0.5, label='End Calibration')
    
    plt.title(f'Health Index Trend: 1st_test | Bearing {BEARING_COL//2 + 1}')
    plt.xlabel('File Index')
    plt.ylabel('HI Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Итоги:")
    print(f"   Final HI: {hi_scores[-1]:.4f}")
    print(f"   Max HI: {np.max(hi_scores):.4f}")
    print(f"   Статус: {'⚠️ ОТКАЗ' if hi_scores[-1] > monitor.threshold else '✅ НОРМА'}")
else:
    print("❌ Не удалось загрузить данные. Проверьте путь и формат файлов.")

import matplotlib.pyplot as plt

## --- Визуализация и сохранение ---
if hi_scores:
    import matplotlib.pyplot as plt
    import os

    # 1. Подготовка директории для сохранения
    output_dir = 'assets'  
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Формирование имени файла
    filename = f'hi_trend_col_{BEARING_COL}.png'
    filepath = os.path.join(output_dir, filename)

    # 3. Построение графика
    plt.figure(figsize=(12, 5))
    plt.plot(hi_scores, label='Health Index', color='#8c00ff')
    plt.axhline(y=monitor.threshold, color='r', linestyle='--', 
                label=f'Threshold ({monitor.threshold:.2f})')
    plt.axvline(x=CONFIG['baseline_size'], color='g', linestyle=':', 
                alpha=0.5, label='End Calibration')
    
    # ✅ Настройка пределов осей
    plt.xlim(0, 2000)        # Ось X: от 0 до 2000
    plt.ylim(0, 30)          # Ось Y: от 0 до 25
    '''
    # Автоматический расчет предела Y с запасом 10%
    y_max = max(max(hi_scores), monitor.threshold) * 1.1
    plt.ylim(0, y_max)
    '''
    # Фиксированный предел X по количеству файлов
    plt.xlim(0, len(hi_scores))
    
    plt.title(f'Health Index Trend: 1st_test | Bearing Col {BEARING_COL}')
    plt.xlabel('File Index')
    plt.ylabel('HI Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 4. Сохранение
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"📊 График сохранен в {filepath}")
    except Exception as e:
        print(f"❌ Ошибка сохранения графика: {e}")
    
    # 5. Отображение
    try:
        plt.show()
    except Exception:
        print("⚠️ Не удалось открыть окно графика.")