import mlflow
import joblib
import pandas as pd
from loguru import logger

# Clase encargada de cargar modelos entrenados, generar predicciones y registrar resultados
class ModelPredictor:
    # Inicialización de la clase con configuración y setup de MLflow
    def __init__(self, config):
        self.config = config                                            # Configuración general del predictor
        self.model = None                                               # Modelo que se cargará posteriormente
        mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", ""))  # Configura URI de MLflow

    # Carga del modelo entrenado (Random Forest o XGBoost)
    def load_model(self, model_type=None, model_file: str | None = None):
        """
        Carga el modelo entrenado desde archivo local o desde el registro de MLflow.

        Parámetros:
        - model_type: 'rf' o 'xgb' para seleccionar la ruta configurada si model_file es None.
        - model_file: path o URI a usar (p. ej. 'models:/RFRegressor/1' o ruta local a .pkl).
        """
        if model_file is None:
            if model_type is None:
                model_type = self.config.get("use_model", "rf").lower()

            model_file = (
                self.config["rf_model_file_path"] if model_type == "rf" else self.config["xgb_model_file_path"]
            )

        logger.info(f"Cargando modelo desde {model_file}")

        # Soportar URIs de MLflow Model Registry como models:/... o runs:/...
        try:
            if isinstance(model_file, str) and model_file.startswith(("models:/", "runs:/")):
                logger.info(f"Cargando modelo desde MLflow URI: {model_file}")
                # Cargar como PyFunc para cubrir diversos tipos de modelos
                self.model = mlflow.pyfunc.load_model(model_file)
            else:
                # Intentar cargar con joblib (pickle) como fallback
                try:
                    self.model = joblib.load(model_file)
                except Exception:
                    # última opción: intentar cargar con mlflow (por si es un path local de un artifact)
                    logger.info(f"joblib.load falló; intentando mlflow.pyfunc.load_model({model_file})")
                    self.model = mlflow.pyfunc.load_model(model_file)

            logger.success("Modelo cargado correctamente.")
        except Exception as e:
            logger.exception(f"Error cargando modelo desde {model_file}: {e}")
            raise

    # Generación de predicciones sobre nuevas muestras
    def predict(self, X_new):
        """
        Genera predicciones con el modelo cargado.
        """
        if self.model is None:
            self.load_model()                                                   # Carga modelo si no se ha hecho

        logger.info(f"Generando predicciones sobre {len(X_new)} muestras...")   # Log de inicio
        preds = self.model.predict(X_new)                                       # Predicciones

        # Manejo de predicciones multioutput
        if preds.ndim == 1:
            preds_df = pd.DataFrame(preds, columns=["prediction"])              # DataFrame para salida univariada
        else:
            preds_df = pd.DataFrame(
                preds, columns=[f"target_{i}" for i in range(preds.shape[1])]   # Columnas para multisalida
            )

        logger.success("Predicciones generadas exitosamente.")                  # Log de éxito
        return preds_df

    # Guardado de predicciones y registro en MLflow
    def save_predictions(self, preds_df):
        """
        Guarda las predicciones en disco y las registra en MLflow
        """
        preds_df.to_csv(self.config["output_file"], index=False)                # Guardado CSV
        logger.info(f"Predicciones guardadas en {self.config['output_file']}")  # Log de guardado

        # Configuración del experimento en MLflow
        mlflow.set_experiment("Predictions_Tracking")

        # Registro de artefactos y tags
        with mlflow.start_run(run_name="Prediction_Run"):
            mlflow.log_artifact(str(self.config["output_file"]))                # Guardar archivo en MLflow
            mlflow.set_tags({                                                   # Tags descriptivos
                "stage": "prediction",
                "model_type": self.config.get("use_model", "rf"),
                "data_source": self.config.get("data_source", "unknown"),
            })

    # Flujo completo: carga, predicción, guardado y registro
    def run_prediction(self, X_new):
        """
        Flujo completo: carga modelo, predice, guarda y registra.
        """
        preds_df = self.predict(X_new)  # Genera predicciones
        self.save_predictions(preds_df) # Guarda y registra resultados
        return preds_df                 # Retorna DataFrame con predicciones