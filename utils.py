import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import cfg

# =============================================================================
# 1. Загрузка данных
# =============================================================================

def load_ims_file(filepath):
    """
    Загрузка одного файла IMS.
    Файлы могут быть разделены табуляцией или пробелами, без заголовков.
    Возвращает сигнал выбранного подшипника (cfg.BEARING_COL).
    """
    try:
        # Попытка чтения с разделителем whitespace
        df = pd.read_csv(filepath, sep=r'\s+', header=None, encoding='utf-8', on_bad_lines='skip')
        
        if df.empty:
            return None
            
        col_idx = cfg.BEARING_COL
        if col_idx >= df.shape[1]:
            print(f"⚠️ Warning: Column {col_idx} not found in {filepath}. Shape: {df.shape}")
            return None
            
        signal = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
        signal = signal[np.isfinite(signal)]
        
        # Проверка длины
        if len(signal) != cfg.N_POINTS:
            # Если сигнал короче, можно дополнить нулями или отбросить
            if len(signal) < cfg.N_POINTS:
                return None
            signal = signal[:cfg.N_POINTS]
            
        return signal
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_file_list(directory):
    """
    Получение отсортированного списка файлов с фильтрацией мусора.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    all_files = glob.glob(os.path.join(directory, '*'))
    
    excluded_ext = ('.npy', '.npz', '.png', '.jpg', '.csv', '.zip', '.DS_Store', '.txt')
    
    valid_files = [
        f for f in all_files
        if os.path.isfile(f)
        and not f.endswith(excluded_ext)
        and ':Zone.Identifier' not in f
        and not os.path.basename(f).startswith('.')
    ]
    
    return sorted(valid_files)

# =============================================================================
# 2. Извлечение признаков
# =============================================================================

def extract_features(signal):
    """
    Извлечение признаков из сигнала.
    Returns: dict с признаками
    """
    if signal is None or len(signal) == 0 or not np.isfinite(signal).all():
        return None

    # --- Time Domain ---
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    mean_abs = np.mean(np.abs(signal))
    
    crest_factor = peak / rms if rms > 1e-10 else 0.0
    kurtosis = stats.kurtosis(signal)
    skewness = stats.skew(signal)
    
    # Защита от NaN/Inf
    if not np.isfinite(kurtosis): kurtosis = 0.0
    if not np.isfinite(skewness): skewness = 0.0

    # --- Frequency Domain ---
    fft_vals = np.fft.rfft(signal)
    amplitude = np.abs(fft_vals)
    
    # Энергия в полосах
    n_bands = cfg.N_BANDS
    bands = np.array_split(amplitude, n_bands)
    spectral_features = {f'Band_{i}': np.sum(b**2) for i, b in enumerate(bands)}

    features = {
        'RMS': rms,
        'Peak': peak,
        'CrestFactor': crest_factor,
        'Kurtosis': kurtosis,
        'Skewness': skewness,
        **spectral_features
    }
    
    # Финальная зачистка
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0
            
    return features

def build_feature_matrix(file_list):
    """
    Массовая обработка файлов и создание матрицы признаков.
    """
    data = []
    timestamps = []
    
    print(f"🔧 Extraction: Обработка {len(file_list)} файлов...")
    
    for i, fpath in enumerate(file_list):
        signal = load_ims_file(fpath)
        if signal is None:
            continue
            
        feats = extract_features(signal)
        if feats is None:
            continue
            
        data.append(feats)
        timestamps.append(os.path.basename(fpath))
        
        if (i + 1) % 100 == 0:
            print(f"  Обработано {i+1}/{len(file_list)}...")
    
    df_features = pd.DataFrame(data)
    df_features['Timestamp'] = timestamps
    print(f"✅ Матрица признаков создана: {df_features.shape}")
    return df_features

# =============================================================================
# 3. Визуализация
# =============================================================================

def plot_results(df_features, hi_values, title='Health Index Analysis'):
    """
    Визуализация RMS, Kurtosis и Health Index.
    """
    import matplotlib.pyplot as plt
    plt.style.use(cfg.PLOT_STYLE)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    x_axis = np.arange(len(df_features))
    baseline_end = int(len(df_features) * cfg.BASELINE_FRAC)
    
    # RMS
    axes[0].plot(x_axis, df_features['RMS'], color='steelblue', alpha=0.7)
    axes[0].set_title('RMS (Временная область)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Амплитуда [g]')
    axes[0].axvline(baseline_end, color='r', linestyle='--', alpha=0.5, label='End of Baseline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Kurtosis
    axes[1].plot(x_axis, df_features['Kurtosis'], color='darkorange', alpha=0.7)
    axes[1].set_title('Kurtosis (Импульсность)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Kurtosis')
    axes[1].grid(True, alpha=0.3)
    
    # Health Index
    axes[2].plot(x_axis, hi_values, color='#8c00ff', linewidth=2)
    axes[2].set_title('Health Index', fontsize=12, fontweight='bold', color='#8c00ff')
    axes[2].set_ylabel('HI Score')
    axes[2].set_xlabel('Временной индекс')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()