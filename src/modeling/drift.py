# drift.py
import typer
from src.config.config import conf
from src.utils import paths
from pathlib import Path
from src.utils.mlflow_client import MLFlowClient
from src.modeling.predict_model import ModelPredictor
from src.modeling.drift_detection import DriftDetection
import requests

app = typer.Typer()

# Comando principal para generar drift
@app.command()
def main():
    # Asegura que exista la ruta de salida intermedia
    paths.ensure_path(conf.paths.processed)  # Crea la carpeta si no existe
    paths.ensure_path(conf.paths.synthetic)  # Crea la carpeta si no existe

    # Inicialización del objeto ModelPredictor con configuración
    # Intentar obtener los latest models desde MLflow Registry si está disponible
    rf_model_uri = ""
    xgb_model_uri = ""

    try:
        ml_client = MLFlowClient(conf.training.mlflow_tracking_uri)
        try:
            ml_client.check_remote_available()

            rf_latest = ml_client.get_latest_version(conf.training.rf_registry_model_name)
            xgb_latest = ml_client.get_latest_version(conf.training.xgb_registry_model_name)

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

    if not rf_model_uri:
        model_path_last_rev = paths.get_latest_version_path(conf.training.rf_model_path)
        rf_model_uri = model_path_last_rev / conf.training.rf_model_file

    if not xgb_model_uri:
        model_path_last_rev = paths.get_latest_version_path(conf.training.xgb_model_path)
        xgb_model_uri = model_path_last_rev / conf.training.xgb_model_file

    try:
        config = {
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,
            "rf_model_file_path": rf_model_uri,
            "xgb_model_file_path": xgb_model_uri,
            "use_model": conf.prediction.use_model,
        }

        predictor = ModelPredictor(config=config)
        
        rf_model = predictor.load_model(model_file=rf_model_uri)  
        xgb_model = predictor.load_model(model_file=xgb_model_uri)  
    except Exception:
        raise RuntimeError("No se pudo inicializar el ModelPredictor.")

    if conf.prediction.use_model == "rf":
        model = rf_model
    else:
        model = xgb_model
    
    drift = DriftDetection(
        X_path=conf.data.processed_data.x_test_file,    
        y_path=conf.data.processed_data.y_test_file,
        synthetic_data_source=conf.data.synthetic_data_file,
        model = model
    )
    
    drift.run()  # Carga, limpia, convierte, imputa y guarda el dataset


if __name__ == "__main__":
    app()  # Lanza la aplicación Typer
