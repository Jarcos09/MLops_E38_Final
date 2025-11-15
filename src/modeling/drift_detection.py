# drift_detection.py
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
import pandas as pd
import numpy as np
import pandas as pd


class DriftDetection:
    def __init__(self, X_path: Path, y_path: Path, synthetic_data_path: Path, config: dict):
        self.Xtrain_path = X_path
        self.ytrain_path = y_path
        self.synthetic_data_path=synthetic_data_path
        self.config = config


    def load_dataset(self):
        logger.info(f"Cargando dataset desde: {self.Xtrain_path} y {self.ytrain_path}")
        self.X_train = pd.read_csv(self.Xtrain_path)
        self.y_train = pd.read_csv(self.ytrain_path)
        self.X_drift = pd.read_csv(self.synthetic_data_path)
        self.X_drift = self.X_drift.loc[self.y_train.index]
        

    def run(self):
        self.load_dataset()
        self._classify_columns_and_select_drift()

    # ---------- Clasificación y selección de columnas ----------
    def _classify_columns_and_select_drift(self):
        logger.info(f"Cargando modelo desde: {self.config["use_model"]}")

        if self.config["use_model"] == "rf":
            model = self.config["rf_model"]
        elif self.config["use_model"] == "xgb":
            model = self.config["xgb_model"]

        # === Evaluación sobre conjunto BASE (X_test / y_test) ===
        logger.info("Evaluando modelo")
        y_pred_base = model.predict(self.X_train)
        y_true_base = self.y_train.values if isinstance(self.y_train, pd.DataFrame) else self.y_train

        mse_base = mean_squared_error(y_true_base, y_pred_base)
        mae_base = mean_absolute_error(y_true_base, y_pred_base)
        r2_base = r2_score(y_true_base, y_pred_base)

        # === Evaluación sobre conjunto DRIFT (X_drift / y_drift) ===
        y_pred_drift = model.predict(self.X_drift)

        mse_drift = mean_squared_error(y_true_base, y_pred_drift)
        mae_drift = mean_absolute_error(y_true_base, y_pred_drift)
        r2_drift = r2_score(y_true_base, y_pred_drift)

        # === Comparación de métricas ===
        metrics_df = pd.DataFrame({
            "Dataset": ["Baseline (Test)", "Drift (Sintético)"],
            "MSE": [mse_base, mse_drift],
            "MAE": [mae_base, mae_drift],
            "R²": [r2_base, r2_drift]
        })

        logger.info("Resultados de desempeño del modelo ===")
        print(metrics_df)

        # === Alerta básica de pérdida de desempeño ===
        R2_THRESHOLD = 0.05   # pérdida de R² > 5%
        MSE_INCREASE_THRESHOLD = 0.10  # aumento de MSE > 10%

        r2_drop = r2_base - r2_drift
        mse_increase = (mse_drift - mse_base) / mse_base if mse_base != 0 else 0

        alertas = []
        if r2_drop > R2_THRESHOLD:
            alertas.append(f"Pérdida significativa de R²: {r2_drop:.3f}")
        if mse_increase > MSE_INCREASE_THRESHOLD:
            alertas.append(f"Aumento en MSE de {mse_increase*100:.1f}%")

        if alertas:
            print("\n⚠️ ALERTA DE PÉRDIDA DE DESEMPEÑO DETECTADA:")
            for a in alertas:
                print(" -", a)
        else:
            print("\n✅ Desempeño del modelo estable. No se detecta degradación significativa.")
        

        # ---------- Pruebas estadísticas de drift ----------

        ALPHA = 0.05            # umbral de significancia (p < ALPHA -> drift)
        USE_BONFERRONI = False  # si True, ajusta ALPHA dividiendo por #tests por tipo. (desactivada por defecto)

        # Identificar tipos de columnas
        num_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # (Por seguridad) Asegurar mismas columnas en ambos datasets
        common_cols = [c for c in self.X_train.columns if c in self.X_drift.columns]
        num_cols = [c for c in num_cols if c in common_cols]
        cat_cols = [c for c in cat_cols if c in common_cols]

        results = []

        # --- Kolmogorov–Smirnov (KS) para numéricas ---
        alpha_num = ALPHA / max(1, len(num_cols)) if USE_BONFERRONI else ALPHA
        for c in num_cols:
            a = self.X_train[c].dropna().values
            b = self.X_drift[c].dropna().values
            # Condiciones mínimas
            if len(a) < 2 or len(b) < 2:
                stat, p = np.nan, np.nan
            else:
                stat, p = ks_2samp(a, b, alternative='two-sided', mode='auto')

            results.append({
                "feature": c,
                "type": "numeric",
                "test": "KS",
                "statistic": stat,
                "p_value": p,
                "alpha": alpha_num,
                "drift_detected": (p < alpha_num) if np.isfinite(p) else False,
                "missing_frac_base": self.X_train[c].isna().mean(),
                "missing_frac_drift": self.X_drift[c].isna().mean(),
                "n_unique_base": self.X_train[c].nunique(dropna=True),
                "n_unique_drift": self.X_drift[c].nunique(dropna=True),
            })

        # --- Chi-cuadrado (Chi²) para categóricas ---
        alpha_cat = ALPHA / max(1, len(cat_cols)) if USE_BONFERRONI else ALPHA
        for c in cat_cols:
            # Tablas de frecuencias
            base_counts = self.X_train[c].astype("category")
            drift_counts = self.X_drift[c].astype("category")
            # Alinear categorías (unión de categorías)
            cats = list(set(base_counts.cat.categories).union(set(drift_counts.cat.categories)))
            base_ct = base_counts.cat.set_categories(cats)
            drift_ct = drift_counts.cat.set_categories(cats)

            base_freq = base_ct.value_counts(sort=False)
            drift_freq = drift_ct.value_counts(sort=False)
            contingency = np.vstack([base_freq.values, drift_freq.values])

            # Requisitos: al menos 2 categorías con conteo > 0
            if (contingency.sum(axis=0) > 0).sum() < 2:
                chi2, p, dof = np.nan, np.nan, np.nan
            else:
                chi2, p, dof, _ = chi2_contingency(contingency)

            results.append({
                "feature": c,
                "type": "categorical",
                "test": "Chi2",
                "statistic": chi2,
                "p_value": p,
                "alpha": alpha_cat,
                "drift_detected": (p < alpha_cat) if np.isfinite(p) else False,
                "missing_frac_base": self.X_train[c].isna().mean(),
                "missing_frac_drift": self.X_drift[c].isna().mean(),
                "n_unique_base": int((contingency[0] > 0).sum()) if np.ndim(contingency)==2 else np.nan,
                "n_unique_drift": int((contingency[1] > 0).sum()) if np.ndim(contingency)==2 else np.nan,
            })

        drift_report = pd.DataFrame(results).sort_values(["type", "p_value"], na_position="last").reset_index(drop=True)

        print(f"[INFO] KS aplicado a {len(num_cols)} columna(s) numérica(s).")
        print(f"[INFO] Chi² aplicado a {len(cat_cols)} columna(s) categórica(s).")
        print(f"[INFO] Umbral alpha (num): {alpha_num} | alpha (cat): {alpha_cat}")

        print("\n=== Resumen de pruebas de drift (p-valor) ===")
        #print(drift_report.head(20))

        # Vista de columnas con drift significativo
        drift_hits = drift_report[drift_report["drift_detected"] == True]

        if len(drift_hits) > 0:
            logger.warning("Drift detectado en las siguientes columnas:")
            for _, row in drift_hits.iterrows():
                logger.warning(f" - {row['feature']} | Tipo: {row['type']} | Test: {row['test']} | p-valor: {row['p_value']:.4e}")
        else:
            logger.info("No se detectó drift significativo en ninguna columna.")

        ## 4.3 Identificar Columnas con Drift Significativo

        # Parámetros
        TOP_K = 15  # para mostrar las columnas más afectadas

        # Reglas de severidad:
        # - Numéricas (KS): usamos el estadístico KS (0–1)
        # - Categóricas (Chi²): usamos p-value (inverso para severidad)
        def classify_severity(row):
            if row["type"] == "numeric":
                ks = row["statistic"]
                if not np.isfinite(ks): 
                    return "unknown"
                if ks >= 0.30: return "severe"
                if ks >= 0.20: return "moderate"
                if ks >= 0.10: return "mild"
                return "low"
            else:  # categorical
                p = row["p_value"]
                if not np.isfinite(p):
                    return "unknown"
                if p < 1e-6: return "severe"
                if p < 1e-4: return "moderate"
                if p < 1e-2: return "mild"
                return "low"

        drift_report = drift_report.copy()
        drift_report["severity"] = drift_report.apply(classify_severity, axis=1)

        # Filtrar solo columnas con drift (p < alpha ya calculado en Paso 4.2)
        drift_hits = drift_report[drift_report["drift_detected"] == True].copy()

        # Ranking: numéricas por KS desc, categóricas por p asc
        numeric_hits = drift_hits[drift_hits["type"] == "numeric"].sort_values(
            by=["statistic"], ascending=False
        )
        categorical_hits = drift_hits[drift_hits["type"] == "categorical"].sort_values(
            by=["p_value"], ascending=True
        )

        score = []
        for i, r in drift_hits.iterrows():
            if r["type"] == "numeric":
                s = r["statistic"]  # KS
            else:
                p = r["p_value"] if np.isfinite(r["p_value"]) and r["p_value"]>0 else 1e-300
                s = -np.log10(p)
            score.append(s)
        drift_hits["drift_score"] = score

        top_hits = drift_hits.sort_values(by="drift_score", ascending=False).head(TOP_K)

        print(f"[INFO] Columnas con drift significativo: {len(drift_hits)}")

        ### 4.4 Unificar los Resultados (métricas + pruebas estadísticas)

        

