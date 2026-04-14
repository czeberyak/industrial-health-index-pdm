import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from typing import Dict, Optional, Any, List

class AdaptiveHealthMonitor:
    """
    Промышленный монитор здоровья подшипников с автоматической калибровкой.

    Особенности:
    - State Machine: INIT -> CALIBRATING -> OPERATIONAL.
    - Auto-Calibration: Обучение на первых K записях номинального режима.
    - Health Index: Рассчитывается как модуль проекции на PC1 (HI = |PC1|).
      Это гарантирует HI >= 0 и физический смысл "расстояние от нормы".
    - Domain Adaptation: Автоматическая настройка порога под локальную базовую линию.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация монитора.

        Args:
            config: Словарь с параметрами:
                - baseline_size: int, количество сэмплов для калибровки.
                - n_bands: int, количество спектральных полос.
                - k_sigma: float, коэффициент для порога (mean + k*std).
        """
        self.cfg = config
        self.state = 'INIT'
        self.buffer: List[Dict] = []

        # Трансформеры
        self.scaler: Optional[StandardScaler] = None
        self.reducer: Optional[PCA] = None

        # Статистики порога
        self.hi_baseline_mean: Optional[float] = None
        self.hi_baseline_std: Optional[float] = None
        self.threshold: Optional[float] = None

        # Метаданные
        self.feature_cols: Optional[List[str]] = None
        self.calibration_info: Dict[str, Any] = {}

    def process(self, signal: np.ndarray, timestamp: Optional[str] = None) -> Dict:
        """
        Основной метод обработки сигнала.

        Args:
            signal: np.array, сырой сигнал вибрации.
            timestamp: Optional[str], метка времени для логирования.

        Returns:
            dict с результатами обработки.
        """
        features = self._extract_features(signal)

        # 🔧 Защита от битых сигналов
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

    def _extract_features(self, signal: np.ndarray) -> Optional[Dict]:
        """
        Извлечение признаков с защитой от NaN/Inf.

        Returns:
            dict с признаками или None, если сигнал некорректен.
        """
        if not np.isfinite(signal).all() or len(signal) == 0:
            return None

        # Time Domain
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        crest = peak / rms if rms > 1e-10 else 0.0
        kurt = stats.kurtosis(signal)
        if not np.isfinite(kurt):
            kurt = 0.0

        # Frequency Domain
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

        # Финальная зачистка
        for k, v in features.items():
            if not np.isfinite(v):
                features[k] = 0.0

        return features

    def _calibration_step(self, features: Dict, timestamp: Optional[str]) -> Dict:
        """
        Накопление данных и обучение на базовой линии.
        """
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
        """
        Обучение трансформеров и расчет порога.
        """
        X = pd.DataFrame(self.buffer)[self.feature_cols].values

        # 1. Масштабирование
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 2. PCA
        self.reducer = PCA(n_components=1)
        self.reducer.fit(X_scaled)

        # 3. Расчет HI как модуля проекции
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: HI = |PC1| гарантирует неотрицательность
        X_pca = self.reducer.transform(X_scaled).ravel()
        hi_scores = np.abs(X_pca)

        # 4. Расчет порога на распределении модулей
        self.hi_baseline_mean = float(hi_scores.mean())
        self.hi_baseline_std = float(hi_scores.std())

        k_sigma = self.cfg.get('k_sigma', 3.0)
        self.threshold = self.hi_baseline_mean + k_sigma * self.hi_baseline_std

    def _inference_step(self, features: Dict, timestamp: Optional[str]) -> Dict:
        """
        Инференс на новых данных.
        """
        X = pd.DataFrame([features])[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Получаем проекцию и берем модуль
        projection = self.reducer.transform(X_scaled)
        hi_raw = np.abs(projection.item())

        # Проверка аномалии
        is_anomaly = hi_raw > self.threshold

        return {
            'status': 'OPERATIONAL',
            'hi_score': hi_raw,
            'threshold': self.threshold,
            'is_anomaly': is_anomaly,
            'margin': hi_raw - self.threshold,
            'timestamp': timestamp
        }

    def save(self, path: str):
        """
        Сохранение состояния модели.
        """
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
    def load(cls, path: str) -> 'AdaptiveHealthMonitor':
        """
        Загрузка модели.
        """
        data = joblib.load(path)
        instance = cls(data['cfg'])
        for k, v in data.items():
            setattr(instance, k, v)
        return instance