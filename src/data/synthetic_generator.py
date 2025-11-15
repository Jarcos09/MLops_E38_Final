# src/data/synthetic_generator.py
import joblib
import numpy as np
import pandas as pd
from loguru import logger


class SyntheticDataGenerator:
    def __init__(self, gmm_file: str, n_samples: int, output_path: str):
        self.gmm_file = gmm_file
        self.n_samples = n_samples
        self.output_path = output_path

        logger.debug(f"GMM file: {self.gmm_file}")
        logger.debug(f"Samples to generate: {self.n_samples}")
        logger.debug(f"Output path: {self.output_path}")

        self.gmm = None
        self.feature_names = None

    def load_gmm(self):
        """
        Carga el archivo GMM que contiene:
        - model: GaussianMixture
        - feature_names: columnas utilizadas en el entrenamiento
        """
        logger.info("Cargando modelo GMM...")

        if not self.gmm_file:
            raise ValueError("La ruta del archivo GMM no está especificada.")

        payload = joblib.load(self.gmm_file)

        if "model" not in payload:
            raise KeyError("El archivo GMM no contiene la clave 'model'.")

        self.gmm = payload["model"]
        self.feature_names = payload.get("feature_names")

        logger.success(f"GMM cargado correctamente desde: {self.gmm_file}")


    def generate(self) -> pd.DataFrame:
        """
        Genera datos sintéticos utilizando el modelo GMM cargado.
        """
        if self.gmm is None:
            raise ValueError("Debes cargar el GMM antes de generar datos.")

        logger.info(f"Generando {self.n_samples} datos sintéticos...")

        # GaussianMixture.sample() retorna (X, y)
        X, _ = self.gmm.sample(self.n_samples)

        # Crear DataFrame con nombres de columnas si están disponibles
        if (
            self.feature_names is not None
            and isinstance(self.feature_names, (list, tuple, np.ndarray))
            and len(self.feature_names) == X.shape[1]
        ):
            logger.debug(f"feature_names type: {type(self.feature_names)}")
            df = pd.DataFrame(X, columns=list(self.feature_names))
        else:
            df = pd.DataFrame(X)
            logger.warning(
                "feature_names inválidos o no coinciden con las dimensiones; usando columnas numéricas."
            )

        logger.success(f"{len(df)} filas generadas exitosamente.")
        return df

    def save(self, df: pd.DataFrame):
        df.to_csv(self.output_path, index=False)
        logger.success(f"Datos sintéticos guardados en: {self.output_path}")


    def run(self):
        self.load_gmm()
        df = self.generate()
        self.save(df)