import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_industrial_dashboard(hi_series: np.ndarray, thresholds: dict, nmf_weights: np.ndarray):
    """Отрисовка финального состояния Health Index с зонами алармов."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # График Health Index
    ax1.plot(hi_series, label='Health Index (Projected)', color='blue')
    ax1.axhline(thresholds['warning'], color='orange', linestyle='--', label='Warning (3σ)')
    ax1.axhline(thresholds['alarm'], color='red', linestyle='-', label='Critical Alarm (10σ)')
    
    # Определение точки зарождения дефекта
    onset_idx = np.where(hi_series > thresholds['warning'])[0][0]
    ax1.axvline(onset_idx, color='green', alpha=0.3, label='Potential Defect Onset')
    
    ax1.set_title("Health Index Trajectory & Degradation Monitoring")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График NMF компонентов
    ax2.plot(nmf_weights[:, 0], label='Component 1 (Normal Friction)')
    ax2.plot(nmf_weights[:, 1], label='Component 2 (Defect Frequency)')
    ax2.set_title("NMF Physical Components Evolution")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
