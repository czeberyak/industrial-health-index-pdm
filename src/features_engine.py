import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
from datetime import datetime
from typing import Dict, List, Union
from tqdm.notebook import tqdm  # Используем tqdm.notebook для красивых баров в Jupyter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


'''
Функция get_signal_stats: Она берет массив «сырых» данных вибрации и вычисляет 12 различных математических характеристик (признаков).  

Временные признаки: RMS (мощность), Kurtosis (наличие резких ударов), Crest factor (пиковость) и другие.  

Частотные признаки: С помощью преобразования Фурье (rfft) она смотрит, на каких частотах «звенит» подшипник (низкие, средние или высокие частоты).  

Функция extract_advanced_features: Это цикл, который заходит в каждый файл из папки с данными, читает его, отправляет данные в первую функцию и собирает результаты в одну большую таблицу (DataFrame).  

Блок расчета Ratio: В конце код создает относительный показатель износа, сравнивая вибрацию каждого подшипника со средним значением по группе.
'''

'''
Инструкции по использованию
Настройка частоты: В функции get_signal_stats параметр fs=20000 (20 кГц) должен строго соответствовать настройкам твоего оборудования.  

Путь к данным: Убедись, что переменная data_path указывает на папку, где лежат текстовые файлы NASA (например, 2nd_test).  

Библиотеки: Тебе нужно установить scipy и tqdm (pip install scipy tqdm), иначе код выдаст ошибку.
'''
# --- 1. ФУНКЦИИ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ ---

def get_signal_stats(sig: np.ndarray, fs: int = 20000) -> Dict[str, float]:
    """Считает временные и частотные признаки для одного канала."""
    stats = {}
    
    # Временные признаки
    rms = np.sqrt(np.mean(sig**2))
    stats['rms'] = rms
    stats['mean'] = np.mean(sig)
    stats['var'] = np.var(sig)
    stats['kurt'] = kurtosis(sig, fisher=False)
    stats['skew'] = skew(sig)
    stats['crest'] = np.max(np.abs(sig)) / rms if rms > 1e-9 else 0.0
    stats['p2p'] = np.ptp(sig)
    stats['energy'] = np.sum(sig**2)
    
    # Частотные признаки (FFT)
    n = len(sig)
    abs_fft = np.abs(rfft(sig))
    freqs = rfftfreq(n, 1/fs)
    sum_fft = np.sum(abs_fft)
    
    stats['dom_freq'] = freqs[np.argmax(abs_fft)]
    stats['spec_centroid'] = np.sum(freqs * abs_fft) / sum_fft if sum_fft > 1e-9 else 0.0
    
    stats['band_0_2k'] = np.sum(abs_fft[(freqs >= 0) & (freqs < 2000)])
    stats['band_2_5k'] = np.sum(abs_fft[(freqs >= 2000) & (freqs < 5000)])
    stats['band_5_10k'] = np.sum(abs_fft[(freqs >= 5000) & (freqs <= 10000)])
    
    return stats

def extract_advanced_features(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Обходит директорию с сырыми файлами и формирует датафрейм признаков."""
    data_path = Path(data_dir)
    files = sorted([f for f in data_path.iterdir() if f.is_file() and not f.name.startswith('.')])
    
    if not files:
        raise FileNotFoundError(f"В папке {data_path} не найдено файлов! Проверь путь.")
        
    features: List[Dict[str, float]] =[]
    
    for file_path in tqdm(files, desc="📦 Обработка файлов", unit="file"):
        try:
            df = pd.read_csv(file_path, sep='\t', header=None)
            dt = datetime.strptime(file_path.name, '%Y.%m.%d.%H.%M.%S')
            file_features = {'timestamp': dt}
            
            for i in range(df.shape[1]):
                sig = df[i].values
                sig_stats = get_signal_stats(sig)
                prefix = f'b{i+1}_'
                for stat_name, value in sig_stats.items():
                    file_features[prefix + stat_name] = value
                
            features.append(file_features)
        except Exception as e:
            pass # Игнорируем битые файлы для чистоты вывода
            
    res_df = pd.DataFrame(features).sort_values('timestamp').reset_index(drop=True)
    return res_df

# --- 2. ЗАПУСК И ВИЗУАЛИЗАЦИЯ ---

data_path = Path('../data/2nd_test') 

print(f"Начинаем извлечение признаков из: {data_path.absolute()}")
df_features = extract_advanced_features(data_path)
print(f"Готово! Извлечено признаков для {len(df_features)} файлов.")

# --- 3. ДОБАВЛЕНИЕ СИНТЕТИЧЕСКОГО ПРИЗНАКА b1_rms_ratio ---

# Рассчитываем медиану RMS для каждого момента времени
df_features['rms_median'] = df_features[['b1_rms', 'b2_rms', 'b3_rms', 'b4_rms']].median(axis=1)

# --- Вставь определение функции перед разделом запуска ---

def convert_to_operating_time(df, threshold=0.02):
    """Схлопывает простои и пересчитывает время в моточасы."""
    # Оставляем только рабочие интервалы
    df_active = df[df['rms_median'] > threshold].copy()
    
    # Считаем разницу во времени между файлами
    df_active['time_diff'] = df_active['timestamp'].diff().dt.total_seconds().fillna(600)
    
    # Если разрыв > 1 часа, считаем его как 10-минутный шаг
    df_active.loc[df_active['time_diff'] > 3600, 'time_diff'] = 600
    
    # Накапливаем часы наработки
    df_active['op_hours'] = df_active['time_diff'].cumsum() / 3600.0
    return df_active

# --- В блоке исполнения (внизу файла) ---

# 1. Извлекаем признаки
df_features = extract_advanced_features(data_path)

# 2. Сначала считаем rms_median (он нужен для фильтрации простоя)
rms_cols = [c for c in df_features.columns if 'rms' in c]
df_features['rms_median'] = df_features[rms_cols].median(axis=1)

# 3. ПРИМЕНЯЕМ ПЕРЕСЧЕТ ВРЕМЕНИ (ВСТАВИТЬ ТУТ)
df_features = convert_to_operating_time(df_features)

# 4. Теперь считаем остальные индексы уже на очищенных данных
df_features['b1_rms_ratio'] = df_features['b1_rms'] / df_features['rms_median']

# Создаём отношения для всех подшипников
for i in range(1, 5):
    col_name = f'b{i}_rms_ratio'
    df_features[col_name] = df_features[f'b{i}_rms'] / df_features['rms_median']



