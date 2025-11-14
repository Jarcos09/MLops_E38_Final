# tests/test_predict_model.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.modeling.predict_model import ModelPredictor


# ------------------------------------------------------------------
# Utilidad: config base usada en varias pruebas
# ------------------------------------------------------------------
def base_cfg(tmp_path):
    return {
        "mlflow_tracking_uri": "file:///tmp/mlruns",
        "use_model": "rf",
        "rf_model_file_path": str(tmp_path / "rf.pkl"),
        "xgb_model_file_path": str(tmp_path / "xgb.pkl"),
        "output_file": str(tmp_path / "preds.csv"),
        "data_source": "unit-test",
    }


# ------------------------------------------------------------------
# __init__: set_tracking_uri
# ------------------------------------------------------------------
@pytest.mark.unit
@patch("src.modeling.predict_model.mlflow")
def test_init_sets_tracking_uri(mock_mlflow, tmp_path):
    cfg = base_cfg(tmp_path)
    _ = ModelPredictor(cfg)
    mock_mlflow.set_tracking_uri.assert_called_once_with(cfg["mlflow_tracking_uri"])


# ------------------------------------------------------------------
# load_model: ruta local vía joblib.load
# ------------------------------------------------------------------
@pytest.mark.unit
@patch("src.modeling.predict_model.joblib")
@patch("src.modeling.predict_model.mlflow")
def test_load_model_local_joblib(mock_mlflow, mock_joblib, tmp_path):
    cfg = base_cfg(tmp_path)
    mp = ModelPredictor(cfg)

    fake_model = MagicMock()
    mock_joblib.load.return_value = fake_model

    mp.load_model(model_type="rf", model_file=None)

    # Debe cargar por joblib (no por mlflow)
    mock_joblib.load.assert_called_once_with(cfg["rf_model_file_path"])
    mock_mlflow.pyfunc.load_model.assert_not_called()
    assert mp.model is fake_model


# ------------------------------------------------------------------
# load_model: MLflow URI -> mlflow.pyfunc.load_model
# ------------------------------------------------------------------
@pytest.mark.unit
@patch("src.modeling.predict_model.joblib")
@patch("src.modeling.predict_model.mlflow")
def test_load_model_mlflow_uri(mock_mlflow, mock_joblib, tmp_path):
    cfg = base_cfg(tmp_path)
    mp = ModelPredictor(cfg)

    fake_model = MagicMock()
    mock_mlflow.pyfunc.load_model.return_value = fake_model

    mp.load_model(model_file="models:/SomeModel/1")

    mock_mlflow.pyfunc.load_model.assert_called_once_with("models:/SomeModel/1")
    mock_joblib.load.assert_not_called()
    assert mp.model is fake_model


# ------------------------------------------------------------------
# load_model: fallback a mlflow.pyfunc si joblib falla
# ------------------------------------------------------------------
@pytest.mark.unit
@patch("src.modeling.predict_model.joblib")
@patch("src.modeling.predict_model.mlflow")
def test_load_model_falls_back_to_mlflow_if_joblib_fails(mock_mlflow, mock_joblib, tmp_path):
    cfg = base_cfg(tmp_path)
    mp = ModelPredictor(cfg)

    mock_joblib.load.side_effect = Exception("pickle-fail")
    fake_model = MagicMock()
    mock_mlflow.pyfunc.load_model.return_value = fake_model

    mp.load_model(model_type="xgb", model_file=None)

    mock_joblib.load.assert_called_once_with(cfg["xgb_model_file_path"])
    mock_mlflow.pyfunc.load_model.assert_called_once()  # luego del fallo de joblib
    assert mp.model is fake_model


# ------------------------------------------------------------------
# predict: llama a load_model si no hay modelo; salida 1D → col 'prediction'
# ------------------------------------------------------------------
@pytest.mark.unit
@patch.object(ModelPredictor, "load_model")
def test_predict_calls_load_if_needed_and_returns_df_1d(mock_load, tmp_path):
    cfg = base_cfg(tmp_path)
    mp = ModelPredictor(cfg)

    # Modelo falso con predict que regresa vector 1D
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([0.1, 0.2, 0.3])
    mp.model = None  # fuerza a que llame load_model
    # cuando setee el modelo, que sea nuestro fake
    def set_model_side_effect(*args, **kwargs):
        mp.model = fake_model
    mock_load.side_effect = set_model_side_effect

    X = np.zeros((3, 2))
    out = mp.predict(X)

    mock_load.assert_called_once()
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["prediction"]
    assert len(out) == 3


# ------------------------------------------------------------------
# predict: salida 2D → columnas target_i
# ------------------------------------------------------------------
@pytest.mark.unit
def test_predict_multitarget_returns_df_2d(tmp_path):
    cfg = base_cfg(tmp_path)
    mp = ModelPredictor(cfg)

    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    fake_model = MagicMock()
    fake_model.predict.return_value = preds
    mp.model = fake_model

    out = mp.predict(np.zeros((2, 3)))
    assert list(out.columns) == ["target_0", "target_1"]
    assert out.shape == (2, 2)


# ------------------------------------------------------------------
# save_predictions: escribe CSV y registra en MLflow
# ------------------------------------------------------------------
class _DummyRun:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

@pytest.mark.unit
@patch("src.modeling.predict_model.mlflow")
def test_save_predictions_logs_and_artifacts(mock_mlflow, tmp_path):
    cfg = base_cfg(tmp_path)
    mp = ModelPredictor(cfg)

    # start_run debe comportarse como context manager
    mock_mlflow.start_run.return_value = _DummyRun()

    df = pd.DataFrame({"prediction": [1.0, 2.0]})
    mp.save_predictions(df)

    # archivo creado
    out_path = cfg["output_file"]
    assert pd.read_csv(out_path).shape == (2, 1)

    # llamadas a mlflow
    mock_mlflow.set_experiment.assert_called_once_with("Predictions_Tracking")
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_artifact.assert_called_once_with(out_path)
    # set_tags debe incluir stage/model_type/data_source
    called_tags = mock_mlflow.set_tags.call_args.args[0]
    assert called_tags["stage"] == "prediction"
    assert called_tags["model_type"] == cfg["use_model"]
    assert called_tags["data_source"] == cfg["data_source"]


# ------------------------------------------------------------------
# run_prediction: integra predict + save_predictions y retorna DF
# ------------------------------------------------------------------
@pytest.mark.unit
@patch.object(ModelPredictor, "save_predictions")
@patch.object(ModelPredictor, "predict")
def test_run_prediction_calls_predict_and_save_returns_df(mock_predict, mock_save, tmp_path):
    cfg = base_cfg(tmp_path)
    mp = ModelPredictor(cfg)

    df = pd.DataFrame({"prediction": [0.5, 0.7]})
    mock_predict.return_value = df

    result = mp.run_prediction(np.zeros((2, 1)))

    mock_predict.assert_called_once()
    mock_save.assert_called_once_with(df)
    assert result is df
