import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from pathlib import Path
from typing import Tuple, List

class VibrationProcessor:
    """Класс для обработки сырых вибросигналов и извлечения спектральных признаков."""
    
    def __init__(self, sampling_rate: int = 20480):
        self.fs = sampling_rate

    def load_and_transform(self, file_path: Path, bearing_idx: int = 0) -> np.ndarray:
        """
        Считывает файл, выполняет FFT и возвращает амплитудный спектр.
        Физический смысл: переход от временной реализации к частотному составу,
        где проявляются характерные частоты дефектов (BPFO, BPFI).
        """
        # NASA IMS dataset: 4 columns, no header, tab separated
        df = pd.read_csv(file_path, sep='\t', header=None)
        signal = df.iloc[:, bearing_idx].values
        
        # Центрирование сигнала (удаление DC-составляющей)
        signal = signal - np.mean(signal)
        
        # Быстрое преобразование Фурье (реальная часть)
        n = len(signal)
        amplitudes = np.abs(rfft(signal)) / n
        return amplitudes

    def build_feature_matrix(self, folder_path: str, bearing_idx: int = 0) -> pd.DataFrame:
        """Формирует матрицу X: строки - время (файлы), столбцы - частоты."""
        files = sorted(Path(folder_path).glob('*'))
        spectra = []
        timestamps = []

        for f in files:
            spec = self.load_and_transform(f, bearing_idx)
            spectra.append(spec)
            timestamps.append(f.name)
            
        freqs = rfftfreq(20480, d=1/self.fs)
        return pd.DataFrame(spectra, index=timestamps, columns=freqs)
