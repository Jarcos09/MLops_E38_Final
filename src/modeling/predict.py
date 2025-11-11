# src/predict.py
import typer
import pandas as pd
from src.modeling.predict_model import ModelPredictor
from src.config.config import conf
from src.utils import paths
from src.utils.mlflow_client import MLFlowClient
import requests

app = typer.Typer()

# Comando principal para generar predicciones usando un modelo entrenado
@app.command()
def predict():
    """
    Carga un modelo entrenado y genera predicciones sobre un conjunto de datos.
    Guarda los resultados en un archivo CSV y los registra en MLflow si aplica.
    """
    # Asegura que exista la carpeta para guardar predicciones
    paths.ensure_path(conf.paths.prediction)    # Crea la carpeta si no existe

    # Carga del conjunto de datos de entrada (X_test)
    X_new = pd.read_csv(conf.data.processed_data.x_test_file)   # DataFrame de pruebas

    # Inicialización del objeto ModelPredictor con configuración
    # Intentar obtener los latest models desde MLflow Registry si está disponible
    rf_model_uri = conf.training.rf_model_file
    xgb_model_uri = conf.training.xgb_model_file

    try:
        mlc = MLFlowClient(conf.training.mlflow_tracking_uri)
        try:
            mlc.check_remote_available()

            rf_latest = mlc.get_latest_version(conf.training.rf_registry_model_name)
            xgb_latest = mlc.get_latest_version(conf.training.xgb_registry_model_name)

            if rf_latest and rf_latest.get("version"):
                # usar registry URI para cargar el modelo desde MLflow
                rf_model_uri = paths.build_model_registry_uri(
                    conf.training.rf_registry_model_name, rf_latest["version"]
                )

            if xgb_latest and xgb_latest.get("version"):
                xgb_model_uri = paths.build_model_registry_uri(
                    conf.training.xgb_registry_model_name, xgb_latest["version"]
                )

        except requests.exceptions.RequestException:
            # No está disponible el servidor MLflow remoto; continuar con archivos locales
            pass
    except Exception:
        # Cualquier error inicializando el cliente o consultando, fallback a los archivos locales
        pass

    predictor = ModelPredictor(
        config={
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,   # URI de MLflow
            "rf_model_file": rf_model_uri,               # Ruta modelo RF o registry URI
            "xgb_model_file": xgb_model_uri,             # Ruta modelo XGB o registry URI
            "use_model": conf.prediction.use_model,                     # Modelo a usar
            "output_file": conf.data.prediction_file                    # Archivo de salida
        }
    )

    # Ejecución completa del flujo de predicción
    predictor.run_prediction(X_new)  # Carga modelo, predice, guarda y registra

# Ejecución del CLI cuando se ejecuta el script directamente
if __name__ == "__main__":
    app()  # Lanza la aplicación Typer