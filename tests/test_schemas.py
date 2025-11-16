import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest, PredictionResponse


# ============================================================
#                 TESTS PARA PredictionRequest
# ============================================================

def test_predictionrequest_valid_payload_minimal():
    """Debe aceptar un payload mínimo válido."""
    payload = {
        "instances": [{"f1": 1.0, "f2": 2.0}],
    }

    req = PredictionRequest(**payload)

    assert req.instances == [{"f1": 1.0, "f2": 2.0}]
    # defaults tomados desde conf
    assert req.model_type in {"rf", "xgb"}
    assert isinstance(req.model_version, str)


def test_predictionrequest_valid_with_all_fields():
    """Debe aceptar payload con model_type y model_version explícitos."""
    payload = {
        "instances": [{"f1": 1.0}],
        "model_type": "rf",
        "model_version": "3"
    }

    req = PredictionRequest(**payload)

    assert req.model_type == "rf"
    assert req.model_version == "3"
    assert req.instances == [{"f1": 1.0}]


def test_predictionrequest_invalid_model_type():
    """Debe fallar si model_type no es 'rf' o 'xgb'."""
    payload = {
        "instances": [{"f1": 1.0}],
        "model_type": "svm",
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_predictionrequest_instances_required():
    """Debe fallar si no se pasan instances."""
    payload = {
        "model_type": "rf",
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_predictionrequest_instances_must_be_list():
    """Debe fallar si instances no es lista."""
    payload = {
        "instances": {"a": 1.0}  # incorrecto, no es lista
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


# ============================================================
#                 TESTS PARA PredictionResponse
# ============================================================

def test_predictionresponse_valid():
    """Debe validar una respuesta válida."""
    payload = {
        "predictions": [{"target_0": 1.0, "target_1": 2.0}],
        "data_drift": {"detected": False, "features": []}
    }

    res = PredictionResponse(**payload)

    assert isinstance(res.predictions, list)
    assert isinstance(res.data_drift, dict)
    assert res.predictions[0]["target_0"] == 1.0


def test_predictionresponse_missing_fields():
    """Debe fallar si falta algún campo obligatorio."""
    payload = {
        "predictions": [{"t": 1}]
        # falta data_drift
    }

    with pytest.raises(ValidationError):
        PredictionResponse(**payload)


def test_predictionresponse_empty_predictions():
    """Debe permitir listas vacías si el modelo no predijo nada."""
    payload = {
        "predictions": [],
        "data_drift": {"detected": False, "features": []}
    }

    res = PredictionResponse(**payload)

    assert res.predictions == []
    assert res.data_drift["detected"] is False
