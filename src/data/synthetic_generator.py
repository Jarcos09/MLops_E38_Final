# src/data/synthetic_generator.py
import joblib
import numpy as np
import pandas as pd
from loguru import logger


class SyntheticDataGenerator:
    def __init__(self,  config: dict, output_path: str):
        self.config = config                # Configuración de preprocesamiento
        self.output_path = output_path
        self.gmm = None
        self.feature_names = None

    def load_gmm(self):
        """
        Carga el archivo GMM que contiene:
        - model: GaussianMixture
        - feature_names: columnas utilizadas en el entrenamiento
        """
        logger.info("Cargando modelo GMM...")

        gmm_file = self.config["gmm_file"]

        if not gmm_file:
            raise ValueError("La ruta del archivo GMM no está especificada.")

        payload = joblib.load(gmm_file)

        if "model" not in payload:
            raise KeyError("El archivo GMM no contiene la clave 'model'.")

        self.gmm = payload["model"]
        self.feature_names = payload.get("feature_names")

        logger.success(f"GMM cargado correctamente desde: {gmm_file}")


    def generate(self) -> pd.DataFrame:
        """
        Genera datos sintéticos utilizando el modelo GMM cargado.
        """
        if self.gmm is None:
            raise ValueError("Debes cargar el GMM antes de generar datos.")
        
        n_samples = self.config["n_samples"]

        logger.info(f"Generando {n_samples} datos sintéticos...")

        # GaussianMixture.sample() retorna (X, y)
        X, _ = self.gmm.sample(n_samples)

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


    def _classify_columns(self, df: pd.DataFrame):
        """Clasifica columnas en binarias, lowcard y continuas."""
        def is_binary_dummy(s):
            if not np.issubdtype(s.dtype, np.number):
                return False
            vals = pd.unique(s.dropna())
            return len(vals) <= 2 and set(np.round(vals, 6)).issubset({0.0, 1.0})

        def is_lowcard_numeric(s):
            return np.issubdtype(s.dtype, np.number) and (3 <= s.nunique(dropna=True) <= 15)

        def is_continuous_numeric(s):
            return np.issubdtype(s.dtype, np.number) and (s.nunique(dropna=True) > 15)

        binary_cols = [c for c in df.columns if is_binary_dummy(df[c])]
        lowcard_cols = [c for c in df.columns if is_lowcard_numeric(df[c])]
        cont_cols = [c for c in df.columns if is_continuous_numeric(df[c])]

        return binary_cols, lowcard_cols, cont_cols


    def _apply_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica un drift suave a los datos sintéticos generados."""
        logger.info("Aplicando drift a los datos sintéticos...")

        rng = np.random.default_rng(123)
        X_drift = df.copy()

        # Clasificar columnas
        binary_cols, lowcard_cols, cont_cols = self._classify_columns(df)
        logger.info(f"binarias={len(binary_cols)} | lowcard={len(lowcard_cols)} | continuas={len(cont_cols)}")

        # ---------------- BINARIAS 0/1 ----------------
        gamma_reweight = self.config["gamma_reweight"]

        for c in binary_cols:
            col = df[c].astype(float)
            p = float(col.mean())
            p_prime = (1 - gamma_reweight) * p + gamma_reweight * (1 - p)
            X_drift[c] = rng.binomial(1, p_prime, size=len(df)).astype(col.dtype)

        # ---------------- LOWCARD ----------------
        lowcard_uniform_mix = self.config["lowcard_uniform_mix"]

        for c in lowcard_cols:
            s = df[c]
            counts = s.value_counts(normalize=True)
            cats = counts.index.values
            p = counts.values
            uniform = np.ones_like(p) / len(p)
            q = (1 - lowcard_uniform_mix) * p + lowcard_uniform_mix * uniform
            q /= q.sum()
            X_drift[c] = rng.choice(cats, size=len(s), replace=True, p=q).astype(s.dtype)

        # ---------------- CONTINUAS ----------------
        mean_shift_std = self.config["mean_shift_std"]
        var_scale = self.config["var_scale"]

        for c in cont_cols:
            s = df[c].astype(float)
            mu, sd = s.mean(), s.std(ddof=0)
            if np.isfinite(sd) and sd > 0:
                shifted = s + mean_shift_std * sd
                X_drift[c] = mu + (shifted - mu) * var_scale
            else:
                X_drift[c] = s

        logger.success("Drift aplicado exitosamente.")
        return X_drift


    def save(self, df: pd.DataFrame):
        df.to_csv(self.output_path, index=False)
        logger.success(f"Datos sintéticos guardados en: {self.output_path}")


    def run(self):
        self.load_gmm()
        df = self.generate()

        if self.config.get("apply_drift", False):
            df = self._apply_drift(df)

        self.save(df)