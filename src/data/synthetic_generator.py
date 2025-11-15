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


    def load_gmm(self):
        logger.info("Cargando modelo GMM...")
        self.gmm = joblib.load(self.gmm_file)
        logger.success("GMM cargado correctamente.")


    def generate(self) -> pd.DataFrame:
        if self.gmm is None:
            raise ValueError("Debes cargar el GMM antes de generar datos.")

        logger.info(f"Generando {self.n_samples} datos sintéticos...")

        X, _ = self.gmm.sample(self.n_samples)

        df = pd.DataFrame(X)
        logger.success(f"{len(df)} filas generadas.")

        return df

    def save(self, df: pd.DataFrame):
        df.to_csv(self.output_path, index=False)
        logger.success(f"Datos sintéticos guardados en: {self.output_path}")


    def run(self):
        self.load_gmm()
        df = self.generate()
        self.save(df)