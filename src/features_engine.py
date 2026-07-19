"""
features_engine.py
Извлечение признаков из сырых файлов вибрации NASA IMS Bearing Dataset.

Как библиотека:
    from features_engine import extract_advanced_features
    df = extract_advanced_features("../data/2nd_test")

Как CLI (извлечь и закэшировать признаки в .pkl):
    python src/features_engine.py --data_path ../data/2nd_test

ВАЖНО (см. README, раздел "Известные ограничения"): каждому столбцу исходного
файла присваивается префикс b{i+1}_ по порядку колонок. Для Set 2 и Set 4
(1 канал = 1 подшипник) это соответствие прямое: b1..b4 = подшипники 1-4.
Для Set 1 (2 канала на подшипник — H и V) b1..b8 — это КАНАЛЫ, а не физические
подшипники: b1=B1_H, b2=B1_V, b3=B2_H, b4=B2_V, b5=B3_H, b6=B3_V, b7=B4_H,
b8=B4_V. Перед тем как трактовать "b3"/"b4"/"b6" для Set 1 как физические
подшипники 3, 4 и (несуществующий) 6, свериться с этой картой.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
from tqdm import tqdm

# --- 1. ФУНКЦИИ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ ---


def get_signal_stats(sig: np.ndarray, fs: int = 20_000) -> Dict[str, float]:
    """Считает временные и частотные признаки для одного канала вибрации.

    Параметры
    ---------
    sig : сырой сигнал одного канала одного файла (20 480 точек = 1с при 20 кГц)
    fs  : частота дискретизации, Гц. Должна соответствовать реальным
          настройкам оборудования — для NASA IMS это 20 000 Гц.
    """
    stats: Dict[str, float] = {}

    # Временные признаки
    rms = np.sqrt(np.mean(sig**2))
    stats["rms"] = rms
    stats["mean"] = np.mean(sig)
    stats["var"] = np.var(sig)
    stats["kurt"] = kurtosis(sig, fisher=False)
    stats["skew"] = skew(sig)
    stats["crest"] = np.max(np.abs(sig)) / rms if rms > 1e-9 else 0.0
    stats["p2p"] = np.ptp(sig)
    stats["energy"] = np.sum(sig**2)

    # Частотные признаки (FFT)
    abs_fft = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), 1 / fs)
    sum_fft = np.sum(abs_fft)

    stats["dom_freq"] = freqs[np.argmax(abs_fft)]
    stats["spec_centroid"] = np.sum(freqs * abs_fft) / sum_fft if sum_fft > 1e-9 else 0.0
    stats["band_0_2k"] = np.sum(abs_fft[(freqs >= 0) & (freqs < 2000)])
    stats["band_2_5k"] = np.sum(abs_fft[(freqs >= 2000) & (freqs < 5000)])
    stats["band_5_10k"] = np.sum(abs_fft[(freqs >= 5000) & (freqs <= 10000)])

    return stats


def extract_advanced_features(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Обходит директорию с сырыми файлами NASA IMS и строит таблицу признаков.

    Каждый файл = один момент времени. Каждая колонка исходного файла
    (один канал акселерометра) получает префикс b1_, b2_, ... по порядку.
    См. предупреждение о нумерации каналов/подшипников в докстринге модуля.
    """
    data_path = Path(data_dir)
    files = sorted(
        f for f in data_path.iterdir() if f.is_file() and not f.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"В папке {data_path} не найдено файлов! Проверь путь.")

    rows: List[Dict[str, float]] = []
    for file_path in tqdm(files, desc="📦 Обработка файлов", unit="file"):
        try:
            df = pd.read_csv(file_path, sep="\t", header=None)
            dt = datetime.strptime(file_path.name, "%Y.%m.%d.%H.%M.%S")
            row: Dict[str, float] = {"timestamp": dt}
            for i in range(df.shape[1]):
                prefix = f"b{i + 1}_"
                for stat_name, value in get_signal_stats(df[i].values).items():
                    row[prefix + stat_name] = value
            rows.append(row)
        except Exception:
            continue  # пропускаем повреждённые/нечитаемые файлы

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# --- 2. CLI: ИЗВЛЕЧЬ И ЗАКЭШИРОВАТЬ ПРИЗНАКИ ---


def _build_cache(data_path: Path, out_path: Path) -> None:
    print(f"🧪 Извлекаю признаки из {data_path}...")
    df = extract_advanced_features(data_path)
    df.to_pickle(out_path)
    print(f"✅ Готово: {len(df)} строк -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Извлечение признаков вибрации из сырых файлов NASA IMS Bearing Dataset"
    )
    parser.add_argument(
        "--data_path", required=True, help="Папка с сырыми файлами, например ../data/2nd_test"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Путь для .pkl с результатом (по умолчанию <имя_папки>_features.pkl)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out) if args.out else Path(f"{data_path.name}_features.pkl")
    _build_cache(data_path, out_path)


if __name__ == "__main__":
    main()
