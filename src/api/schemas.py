from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from src.config.config import conf


class PredictionRequest(BaseModel):
    """Payload esperado por POST /predict."""

    instances: List[Dict[str, Any]] = Field(
        ..., description="Lista de instancias a predecir"
    )

    model_type: Optional[str] = Field(
        conf.prediction.use_model,  # valor por defecto dinámico
        description=f"Tipo de modelo a usar ('rf' o 'xgb'). Por defecto '{conf.prediction.use_model}'."
    )

    model_version: Optional[str] = Field(
        conf.prediction.use_version,
        description="Versión del modelo a usar ('1', '2', etc.). Por defecto 'latest'."
    )

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v):
        allowed = {"rf", "xgb"}
        if v not in allowed:
            raise ValueError(f"model_type debe ser uno de: {allowed}")
        return v


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]] = Field(
        ..., description="Lista de predicciones por instancia"
    )