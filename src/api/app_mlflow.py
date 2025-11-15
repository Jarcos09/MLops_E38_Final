import joblib
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from loguru import logger
from src.config.config import conf
from src.data.clean_dataset import DatasetCleaner
from src.modeling.predict_model import ModelPredictor
from src.utils import mlflow_launcher, paths
from src.utils.mlflow_client import MLFlowClient
from src.modeling.drift_detection import DriftDetection
from .schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="ML Model Serving API",
    description="""
    API para servir modelos de Machine Learning del Equipo 38.
    Incluye endpoints para verificar salud del servicio, listar versiones de modelos
    y realizar predicciones sobre instancias de entrada.

    Resuelve los modelos con base en la ruta MLFlow: models:/<name>/<version>
    """,
    version=str(conf.metadata.version),
    contact={
        "name": conf.metadata.author,
        "url": conf.metadata.url,
    },
    license_info={
        "name": conf.metadata.license_name,
        "url": conf.metadata.license_url,
    },
)

# Cargar predictor globalmente para reutilizar la instancia entre peticiones
predictor = None

@app.on_event("startup")
def startup_event():
    """Inicializa y carga el modelo en memoria al iniciar la app."""
    global predictor

    try:
        cfg = {
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,
            "rf_model_file_path": None,
            "xgb_model_file_path": None,
            "use_model": conf.prediction.use_model,
        }

        # Si la URI de tracking apunta a localhost y MLflow no está arriba,
        # intentar arrancar un servidor MLflow local automáticamente (útil en dev).
        try:
            mlflow_launcher.ensure_mlflow_server(conf.training.mlflow_tracking_uri)
        except Exception:
            logger.exception("No se pudo arrancar MLflow automáticamente; continuar de todas formas")

        predictor = ModelPredictor(config=cfg)
        logger.info("Predictor inicializado y modelo cargado en memoria.")
    except Exception as e:
        logger.exception(f"Error cargando el modelo en startup: {e}")


@app.get("/health", summary="Health check", description="Verifica el estado del servicio.")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/models", response_class=PlainTextResponse, summary="Modelos disponibles", description="Lista los modelos disponibles (nombre, versión más reciente y versiones registradas).")
def list_models():
    """
    Lista los modelos registrados en MLflow en formato de tabla de texto.
    Muestra: Nombre del modelo, última versión y versiones disponibles.
    """
    mlflow_uri = conf.training.mlflow_tracking_uri

    try:
        ml_client = MLFlowClient(mlflow_uri)

        # Si el tracking es remoto, comprobar disponibilidad
        try:
            ml_client.check_remote_available()
        except requests.exceptions.RequestException as re:
            logger.exception(f"No se pudo conectar al servidor MLflow en {ml_client.tracking_uri}: {re}")
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo conectar al servidor MLflow en {ml_client.tracking_uri}. Verifica que esté encendido."
            )

        table_text = ml_client.render_models_table()
        return PlainTextResponse(content=table_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error consultando MLflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict", response_model=PredictionResponse, summary="Predicción", description="Dada una lista de instancias devuelve Y1, Y2 predichas.")
def predict(request: PredictionRequest):
    """Endpoint para realizar predicciones.

    Recibe un JSON con `instances: [{feature: value, ...}, ...]` y devuelve
    una lista de objetos con las predicciones.
    """
    model = None
    
    if predictor is None:
        logger.exception("Modelo no cargado.")
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    try:
        # Si el usuario solicitó un tipo de modelo (rf/xgb), mapear al registry y cargar la versión deseada
        if getattr(request, "model_type", None):
            model_type = request.model_type.lower()

            if model_type not in {"rf", "xgb"}:
                logger.exception("Tipo de modelo no valido.")
                raise HTTPException(status_code=400, detail="model_type must be 'rf' or 'xgb'")

            # Mapear el nombre lógico a los nombres del registry
            registry_name = (
                conf.training.rf_registry_model_name if model_type == "rf" else conf.training.xgb_registry_model_name
            )

            # Si el usuario provee version, usarla; si no, intentar obtener la latest desde MLflow
            desired_version = getattr(request, "model_version", None)

            model_uri_to_load = None

            try:
                ml_client = MLFlowClient(conf.training.mlflow_tracking_uri)
                # comprobar disponibilidad remota
                try:
                    ml_client.check_remote_available()

                    if desired_version == conf.prediction.use_version:
                        latest = ml_client.get_latest_version(registry_name)
                        if latest and latest.get("version"):
                            model_uri_to_load = paths.build_model_registry_uri(registry_name, latest.get("version"))
                    else:
                        model_uri_to_load = paths.build_model_registry_uri(registry_name, desired_version)

                except requests.exceptions.RequestException:
                    # MLflow no disponible, fallback a archivos locales
                    logger.exception(f"No se pudo conectar al servidor MLflow en {ml_client.tracking_uri}.")
                    raise HTTPException(
                        status_code=503,
                        detail=f"No se pudo conectar al servidor MLflow en {ml_client.tracking_uri}. Verifica que esté encendido."
                    )
            except Exception:
                logger.exception("Error consultando MLflow para obtener el modelo solicitado.")
                raise HTTPException(
                    status_code=500, 
                    detail="Error consultando MLflow para obtener el modelo solicitado."
                )

            # Si se determinó una URI de registry, cargarla; si no, usar la lógica previa de archivos locales
            if model_uri_to_load is None:
                logger.exception("No se pudo determinar la URI del modelo solicitado.")
                raise HTTPException(
                    status_code=404, 
                    detail=f"No se pudo determinar la URI del modelo solicitado ({model_type} v{desired_version})."
                )

            # En caso de error
            try:
                logger.info(f"Intentando cargar modelo desde MLflow URI: {model_uri_to_load}")
                model = predictor.load_model(model_file=model_uri_to_load)
            except Exception:
                logger.exception("No se pudo cargar el modelo correctamente.")
                raise HTTPException(
                    status_code=404, 
                    detail=f"No se pudo encontrar el modelo solicitado ({model_type} v{desired_version})."
                )

        # `instances` es una lista de dicts (feature->valor)
        X_new = pd.DataFrame(request.instances)

        # Pasar X_new por DatasetCleaner (input en memoria)
        try:
            # asegurar que exista la carpeta interim (igual que el CLI de limpieza)
            try:
                paths.ensure_path(conf.paths.interim)
            except Exception:
                logger.debug("No se pudo asegurar la ruta interim; continuando de todas formas")

            cleaner = DatasetCleaner(
                input_path=X_new,
                output_path=conf.data.interim_data_file,
                skew_threshold=conf.cleaning.skew_threshold,
            )
            X_new = cleaner.run()
        except Exception as e:
            logger.warning(f"DatasetCleaner falló en /predict; usando X_new sin limpiar: {e}")

        # Luego aplicar el preprocesador (OneHotEncoder/ColumnTransformer)
        try:
            # Preferir un preprocessor serializado creado durante training
            preproc_path = Path(conf.preprocessing.preprocessor_file)
            if preproc_path.exists():
                preprocessor = joblib.load(preproc_path)
                logger.info("Preprocessor serializado cargado desde models/preprocessor.pkl")
                # asegurarnos de pasar únicamente las columnas de features esperadas
                feature_cols = list(conf.preprocessing.feature_columns)
                X_to_transform = X_new[feature_cols]
                X_proc = preprocessor.transform(X_to_transform)
                # Si el preprocesador devuelve un array numpy, convertirlo a DataFrame
                # usando los nombres de features generados por el preprocessor cuando sea posible.
                if isinstance(X_proc, np.ndarray):
                    feature_names = None
                    # sklearn >=1.0: ColumnTransformer/Pipeline puede exponer get_feature_names_out
                    try:
                        # preferir pasar las columnas originales si get_feature_names_out acepta argumentos
                        try:
                            feature_names = preprocessor.get_feature_names_out(feature_cols)
                        except Exception:
                            feature_names = preprocessor.get_feature_names_out()
                    except Exception:
                        feature_names = None

                    if feature_names is not None and len(feature_names) == X_proc.shape[1]:
                        X_proc = pd.DataFrame(X_proc, columns=list(feature_names))
                        logger.debug("Converted X_proc numpy array to DataFrame using preprocessor.get_feature_names_out()")
                    else:
                        # No tenemos nombres de columna adecuados; dejar numpy pero advertir
                        logger.warning(
                            "Preprocessor transform returned numpy array but no feature names available; "
                            "this can cause schema mismatch when loading pyfunc models that expect named columns."
                        )
            else:
                logger.info("Preprocessor serializado no encontrado.")
                raise FileNotFoundError("Preprocessor serializado no encontrado.")

            # X_proc es un array numpy listo para pasar al predictor
            X_new = X_proc
        except Exception as e:
            logger.warning(f"Preprocesamiento falló en /predict; pasando X_new sin transformar: {e}")

        preds_df = predictor.predict(X_new)
        result = preds_df.to_dict(orient="records")


        # ============================================================
        #                 EJECUTAR DETECCIÓN DE DRIFT              
        # ============================================================
        try:
            if model is not None:
                drift_detector = DriftDetection(
                    X_path=conf.data.processed_data.x_train_file,
                    y_path=conf.data.processed_data.y_train_file,
                    synthetic_data_source=X_new,
                    model=model
                )

                drift_report = drift_detector.run()

                # --- Ajuste correcto ---
                if drift_report is not None and "severity" in drift_report.columns:
                    
                    drift_cols = drift_report[
                        drift_report["severity"].isin(["moderate", "severe"])
                    ]["feature"].tolist()

                    drift_info = {
                        "detected": len(drift_cols) > 0,
                        "features": drift_cols,
                    }
                else:
                    drift_info = {
                        "detected": False,
                        "features": [],
                    }

        except Exception as e:
            logger.error(f"Drift detection empty at /predict: {e}")
            drift_info = {
                "detected": False,
                "features": [],
            }

        return {
            "predictions": result,
            "data_drift": drift_info
        }
    except Exception as exc:
        logger.exception("Error al generar la predicción")
        raise HTTPException(status_code=500, detail=str(exc))


@app.on_event("shutdown")
def shutdown_event():
    """Intentar detener el MLflow server arrancado por la app en modo dev (si aplica)."""
    try:
        mlflow_launcher.stop_mlflow_server()
    except Exception:
        logger.debug("No se pudo detener MLflow automáticamente o no había servidor gestionado")