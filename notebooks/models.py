from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class DegradationModel:
    """Реализация логики Unsupervised Learning для оценки износа."""
    
    def __init__(self, n_components_nmf: int = 2):
        # PCA с центрированием для выделения вектора деградации
        self.pca_pipe = Pipeline([
            ('scaler', StandardScaler(with_std=False)),
            ('pca', PCA(n_components=1))
        ])
        # NMF для разделения физических источников шума
        self.nmf = NMF(n_components=n_components_nmf, init='nndsvda', random_state=42)
        self.degradation_vector = None

    def fit_degradation_vector(self, X: np.ndarray):
        """
        Рассчитывает вектор деградации.
        Физический смысл: направление в многомерном пространстве частот, 
        вдоль которого происходит максимальное изменение состояния от 'нормы' к 'отказу'.
        """
        self.pca_pipe.fit(X)
        # Вектор деградации — это первая главная компонента
        self.degradation_vector = self.pca_pipe.named_steps['pca'].components_[0]

    def calculate_health_index(self, X: np.ndarray) -> np.ndarray:
        """
        Проекция текущего спектра на вектор деградации (Скалярное произведение).
        HI = 0 (норма), рост HI указывает на развитие дефекта.
        """
        X_centered = X - np.mean(X[:100], axis=0) # Центрирование относительно начала теста
        return np.dot(X_centered, self.degradation_vector)

    def get_nmf_components(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Разложение на базовые профили (норма/износ) и их веса во времени."""
        W = self.nmf.fit_transform(X) # Веса (активность компонент)
        H = self.nmf.components_      # Базовые спектры
        return W, H
