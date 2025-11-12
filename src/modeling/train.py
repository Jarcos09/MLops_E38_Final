# src/train.py
import typer
import pandas as pd
from src.modeling.train_model import ModelTrainer
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

# Comando principal para entrenamiento de modelos Random Forest y XGBoost
@app.command()
def train():
    """
    Ejecuta el flujo completo de entrenamiento:
    1) Carga conjuntos de entrenamiento y prueba (X e y).
    2) Inicializa la clase ModelTrainer con los datos y configuración.
    3) Entrena Random Forest y XGBoost, registrando resultados en MLflow.
    """
    # Asegurar que exista la carpeta para modelos
    paths.ensure_path(conf.paths.models)
    paths.ensure_path(conf.training.rf_model_path)
    paths.ensure_path(conf.training.xgb_model_path)

    # Carga de datasets procesados
    X_train = pd.read_csv(conf.data.processed_data.x_train_file)
    X_test = pd.read_csv(conf.data.processed_data.x_test_file)
    y_train = pd.read_csv(conf.data.processed_data.y_train_file)
    y_test = pd.read_csv(conf.data.processed_data.y_test_file)

    # Inicialización del objeto de entrenamiento con configuración
    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config={
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,
            "random_state": conf.training.random_state,
            "rf_experiment_name": conf.training.rf_experiment_name,
            "rf_registry_model_name": conf.training.rf_registry_model_name,
            "rf_model_path": conf.training.rf_model_path,
            "rf_model_file": conf.training.rf_model_file,
            "xgb_experiment_name": conf.training.xgb_experiment_name,
            "xgb_registry_model_name": conf.training.xgb_registry_model_name,
            "xgb_model_path": conf.training.xgb_model_path,
            "xgb_model_file": conf.training.xgb_model_file
        }
    )

    # Entrenamiento de Random Forest
    trainer.train_random_forest()

    # Entrenamiento de XGBoost
    trainer.train_xgboost()

# Punto de entrada de ejecución del script
if __name__ == "__main__":
    app()