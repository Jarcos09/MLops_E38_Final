import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from loguru import logger
from src.config.config import conf
from src.data.clean_dataset import DatasetCleaner
from src.modeling.predict_model import ModelPredictor
from src.utils import paths
from .schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="ML Model Serving", version=str(conf.metadata.version))

# Cargar predictor globalmente para reutilizar la instancia entre peticiones
predictor = None

@app.on_event("startup")
def startup_event():
    """Inicializa y carga el modelo en memoria al iniciar la app."""
    global predictor
    try:
        model_path_last_rev = paths.get_latest_version_path(conf.training.rf_model_path)
        rf_model_uri = model_path_last_rev / conf.training.rf_model_file

        model_path_last_rev = paths.get_latest_version_path(conf.training.xgb_model_path)
        xgb_model_uri = model_path_last_rev / conf.training.xgb_model_file

        cfg = {
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,
            "rf_model_file_path": rf_model_uri,
            "xgb_model_file_path": xgb_model_uri,
            "use_model": conf.prediction.use_model,
        }

        predictor = ModelPredictor(config=cfg)
        logger.info("Predictor inicializado y modelo cargado en memoria.")
    except Exception as e:
        logger.error(f"Error cargando el modelo en startup: {e}")


@app.get("/health", summary="Health check", description="Verifica el estado del servicio.")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/models", response_class=PlainTextResponse, summary="Modelos disponibles", description="Lista los modelos disponibles (nombre, versión más reciente y versiones registradas).")
def list_models():
    """
    Explora la carpeta local 'models/' y devuelve una tabla de texto con:
      - Nombre del modelo
      - Última versión disponible
      - Todas las versiones encontradas
    """
    models_path = paths.ensure_path(conf.paths.models)

    if not os.path.exists(models_path):
        return "No existe la carpeta 'models/'."
    
    # Tabla para acumular info
    headers = ["Model Name", "Latest Version", "Available Versions"]
    rows = []

    # Recorre los modelos en la carpeta base
    for model_name in sorted(os.listdir(models_path)):
        model_path = os.path.join(models_path, model_name)
        if not os.path.isdir(model_path):
            continue  # saltar archivos sueltos

        # Recolectar versiones (subcarpetas numéricas)
        versions = []
        for v in os.listdir(model_path):
            version_path = os.path.join(model_path, v)
            if os.path.isdir(version_path) and v.isdigit():
                versions.append(int(v))

        if not versions:
            continue

        latest_version = max(versions)
        versions_str = ", ".join(str(v) for v in sorted(versions)) if versions else "—"
        last_version_str = f"{latest_version}" if latest_version else "—"
        rows.append([model_name, str(last_version_str), versions_str])

    if not rows:
        return "No se encontraron modelos versionados en 'models/'."

    # Calcular anchos de columnas
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(3)]

    def fmt_row(row):
        return "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(3)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)

    table_text = "\n".join(lines)
    return table_text


@app.post("/predict", response_model=PredictionResponse, summary="Predicción", description="Dada una lista de instancias devuelve Y1, Y2 predichas.")
def predict(request: PredictionRequest):
    """Endpoint para realizar predicciones.

    Recibe un JSON con `instances: [{feature: value, ...}, ...]` y devuelve
    una lista de objetos con las predicciones.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    try:
        # Si el usuario solicitó un tipo de modelo (rf/xgb), mapear al registry y cargar la versión deseada
        if getattr(request, "model_type", None):
            model_type = request.model_type.lower()

            if model_type not in {"rf", "xgb"}:
                logger.exception("Tipo de modelo no valido.")
                raise HTTPException(status_code=400, detail="model_type must be 'rf' or 'xgb'")
            
            # Si el usuario provee version, usarla; si no, intentar obtener la latest desde MLflow
            desired_version = getattr(request, "model_version", None)

            try:
                if desired_version == conf.prediction.use_version:
                    logger.info("Intentando con la versión más reciente disponible (fallback local).")
                    predictor.load_model(model_type=model_type)
                else:
                    model_path = conf.training.rf_model_path if model_type == "rf" else conf.training.xgb_model_path
                    model_file = conf.training.rf_model_file if model_type == "rf" else conf.training.xgb_model_file
                    model_file_path = paths.build_model_local_path(model_path, desired_version, model_file)
                    logger.info(f"Intentando cargar modelo local desde: {model_file_path}")
                    predictor.load_model(model_file=model_file_path)
            # En caso de error, fallback automático
            except Exception as e:
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

        return {"predictions": result}
    except Exception as exc:
        logger.exception("Error al generar la predicción")
        raise HTTPException(status_code=500, detail=str(exc))