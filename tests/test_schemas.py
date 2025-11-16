import pytest
from pydantic import ValidationError
from src.api.schemas import PredictionRequest, PredictionResponse


# ============================================================
#                 TESTS PARA PredictionRequest
# ============================================================

@pytest.mark.unit
def test_predictionrequest_valid_payload_minimal():
    payload = {
        "instances": [{"f1": 1.0, "f2": 2.0}],
    }

    req = PredictionRequest(**payload)

    assert req.instances == [{"f1": 1.0, "f2": 2.0}]
    assert req.model_type in {"rf", "xgb"}
    assert isinstance(req.model_version, str)


@pytest.mark.unit
def test_predictionrequest_valid_with_all_fields():
    payload = {
        "instances": [{"f1": 1.0}],
        "model_type": "rf",
        "model_version": "3"
    }

    req = PredictionRequest(**payload)

    assert req.model_type == "rf"
    assert req.model_version == "3"
    assert req.instances == [{"f1": 1.0}]


@pytest.mark.unit
def test_predictionrequest_invalid_model_type():
    payload = {
        "instances": [{"f1": 1.0}],
        "model_type": "svm",
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


@pytest.mark.unit
def test_predictionrequest_instances_required():
    payload = {
        "model_type": "rf",
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


@pytest.mark.unit
def test_predictionrequest_instances_must_be_list():
    payload = {
        "instances": {"a": 1.0}  # incorrecto
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


# ============================================================
#                 TESTS PARA PredictionResponse
# ============================================================

@pytest.mark.unit
def test_predictionresponse_valid():
    payload = {
        "predictions": [{"target_0": 1.0, "target_1": 2.0}],
        "data_drift": {"detected": False, "features": []}
    }

    res = PredictionResponse(**payload)

    assert isinstance(res.predictions, list)
    assert isinstance(res.data_drift, dict)
    assert res.predictions[0]["target_0"] == 1.0


@pytest.mark.unit
def test_predictionresponse_missing_fields():
    payload = {
        "predictions": [{"t": 1}]
    }

    with pytest.raises(ValidationError):
        PredictionResponse(**payload)


@pytest.mark.unit
def test_predictionresponse_empty_predictions():
    payload = {
        "predictions": [],
        "data_drift": {"detected": False, "features": []}
    }

    res = PredictionResponse(**payload)

    assert res.predictions == []
    assert res.data_drift["detected"] is False
