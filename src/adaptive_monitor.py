import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class AdaptiveHealthMonitor:
    """
    Промышленный монитор здоровья с автоматической калибровкой.
    
    Health Index рассчитывается как абсолютное отклонение от базовой линии.
    Это гарантирует HI >= 0 и физическую интерпретацию: расстояние от нормы.
    
    Состояния:
    - INIT: Ожидание первого сигнала.
    - CALIBRATING: Накопление базовой линии (номинальный режим).
    - OPERATIONAL: Штатный мониторинг и детекция аномалий.
    """

    def __init__(self, config):
        self.cfg = config
        self.state = 'INIT'
        self.buffer = []
        
        # Трансформеры
        self.scaler = None
        self.reducer = None
        
        # Статистики порога
        self.hi_baseline_mean = None
        self.hi_baseline_std = None
        self.threshold = None
        
        # Метаданные
        self.feature_cols = None
        self.calibration_info = {}

    def process(self, signal, timestamp=None):
        features = self._extract_features(signal)
        
        if features is None:
            return {
                'status': 'ERROR',
                'message': 'Invalid signal or features contain NaN/Inf',
                'hi_score': None,
                'is_anomaly': False
            }
        
        if self.state == 'INIT':
            self.feature_cols = list(features.keys())
            self.state = 'CALIBRATING'
            self.calibration_info['start_time'] = timestamp
            
        if self.state == 'CALIBRATING':
            return self._calibration_step(features, timestamp)
        elif self.state == 'OPERATIONAL':
            return self._inference_step(features, timestamp)
        else:
            raise ValueError(f"Unknown state: {self.state}")

    def _extract_features(self, signal):
        if not np.isfinite(signal).all() or len(signal) == 0:
            return None
        
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        crest = peak / rms if rms > 1e-10 else 0.0
        kurt = stats.kurtosis(signal)
        if not np.isfinite(kurt):
            kurt = 0.0
        
        fft_vals = np.fft.rfft(signal)
        amplitude = np.abs(fft_vals)
        n_bands = self.cfg.get('n_bands', 50)
        bands = np.array_split(amplitude, n_bands)
        
        features = {
            'RMS': rms,
            'CrestFactor': crest,
            'Kurtosis': kurt,
            **{f'Band_{i}': np.sum(b**2) for i, b in enumerate(bands)}
        }
        
        for k, v in features.items():
            if not np.isfinite(v):
                features[k] = 0.0
                
        return features

    def _calibration_step(self, features, timestamp):
        self.buffer.append(features)
        progress = len(self.buffer) / self.cfg['baseline_size']
        
        result = {
            'status': 'CALIBRATING',
            'progress': progress,
            'hi_score': None,
            'is_anomaly': False
        }
        
        if len(self.buffer) >= self.cfg['baseline_size']:
            self._fit_models()
            self.state = 'OPERATIONAL'
            self.calibration_info['end_time'] = timestamp
            result['status'] = 'CALIBRATION_COMPLETE'
            result['threshold'] = self.threshold
            result['message'] = f"Auto-Calibration finished. Threshold: {self.threshold:.4f}"
            self.buffer = []
            
        return result

    def _fit_models(self):
        X = pd.DataFrame(self.buffer)[self.feature_cols].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.reducer = PCA(n_components=1)
        self.reducer.fit(X_scaled)
        
        # 🔧 Ключевое изменение: HI как модуль проекции
        X_pca = self.reducer.transform(X_scaled).ravel()
        hi_scores = np.abs(X_pca)
        
        self.hi_baseline_mean = hi_scores.mean()
        self.hi_baseline_std = hi_scores.std()
        
        k_sigma = self.cfg.get('k_sigma', 3.0)
        self.threshold = self.hi_baseline_mean + k_sigma * self.hi_baseline_std

    def _inference_step(self, features, timestamp):
        X = pd.DataFrame([features])[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        projection = self.reducer.transform(X_scaled)
        hi_raw = np.abs(projection.item())
        
        is_anomaly = hi_raw > self.threshold
        
        return {
            'status': 'OPERATIONAL',
            'hi_score': hi_raw,
            'threshold': float(self.threshold),
            'is_anomaly': is_anomaly,
            'margin': hi_raw - self.threshold,
            'timestamp': timestamp
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'state': self.state,
            'scaler': self.scaler,
            'reducer': self.reducer,
            'threshold': self.threshold,
            'hi_baseline_mean': self.hi_baseline_mean,
            'hi_baseline_std': self.hi_baseline_std,
            'feature_cols': self.feature_cols,
            'calibration_info': self.calibration_info,
            'cfg': self.cfg
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(data['cfg'])
        for k, v in data.items():
            setattr(instance, k, v)
        return instance