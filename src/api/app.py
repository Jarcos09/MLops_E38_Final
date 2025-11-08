import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from src.config.config import conf
from src.modeling.predict_model import ModelPredictor
from .schemas import PredictionRequest, PredictionResponse
from src.data.clean_dataset import DatasetCleaner
from src.utils import paths
import joblib
from pathlib import Path
from src.data.preprocess_data import DataPreprocessor


app = FastAPI(title="ML Model Serving", version=str(conf.metadata.version))

# Cargar predictor globalmente para reutilizar la instancia entre peticiones
predictor = None

@app.on_event("startup")
def startup_event():
    """Inicializa y carga el modelo en memoria al iniciar la app."""
    global predictor
    try:
        cfg = {
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,
            "rf_model_file": conf.training.rf_model_file,
            "xgb_model_file": conf.training.xgb_model_file,
            "use_model": conf.prediction.use_model,
        }
        predictor = ModelPredictor(config=cfg)
        logger.info("Predictor inicializado y modelo cargado en memoria.")
    except Exception as e:
        logger.error(f"Error cargando el modelo en startup: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Endpoint para realizar predicciones.

    Recibe un JSON con `instances: [{feature: value, ...}, ...]` y devuelve
    una lista de objetos con las predicciones.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    try:
        # Si el usuario solicitó un tipo de modelo, recargarlo (rf/xgb)
        if getattr(request, "model_type", None):
            model_type = request.model_type.lower()

            if model_type not in {"rf", "xgb"}:
                raise HTTPException(status_code=400, detail="model_type must be 'rf' or 'xgb'")
            
            predictor.load_model(model_type=model_type)

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
