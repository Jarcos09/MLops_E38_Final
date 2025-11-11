from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Payload esperado por POST /predict.

    Usamos una lista de dicts para `instances` para mantener compatibilidad tanto
    con Pydantic v1 como v2 y evitar utilizar `__root__` o `RootModel`.
    """

    instances: List[Dict[str, Any]] = Field(..., description="Lista de instancias a predecir")
    model_type: Optional[str] = Field(
        None, description="Tipo de modelo a usar ('rf' o 'xgb'). Si no se provee, se usará el por defecto en la config."
    )
    model_version: Optional[str] = Field(
        None, description="Versión del modelo a usar en el Model Registry (ej: '1'). Si no se indica, se usará la latest si se puede consultar MLflow."
    )

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]] = Field(..., description="Lista de predicciones por instancia")
