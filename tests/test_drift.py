# tests/test_drift.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.modeling import drift as drift_mod   # ajusta el path si tu módulo drift.py está en otro paquete


@pytest.mark.unit
@patch("src.modeling.drift.DriftDetection")
@patch("src.modeling.drift.ModelPredictor")
@patch("src.modeling.drift.MLFlowClient")
@patch("src.modeling.drift.paths.get_latest_version_path")
@patch("src.modeling.drift.paths.ensure_path")
def test_main_ok(
    mock_ensure_path,
    mock_get_latest_version_path,
    mock_mlflow_client_cls,
    mock_predictor_cls,
    mock_drift_cls,
):
    """
    Test unitario de la función main en drift.py.

    Verifica que:
    - Se llamen ensure_path sobre las rutas processed y synthetic.
    - Se inicialice ModelPredictor y se llamen load_model para rf y xgb.
    - Se cree DriftDetection con un modelo y se ejecute run().
    """

    # --- Mocks de paths ---
    mock_get_latest_version_path.side_effect = lambda base: Path("models/1")

    # --- Mock de MLFlowClient para que NO haga llamadas reales ---
    ml_client_inst = MagicMock()
    # forzamos que check_remote_available falle para que el código use el fallback local
    ml_client_inst.check_remote_available.side_effect = Exception("MLflow not available")
    mock_mlflow_client_cls.return_value = ml_client_inst

    # --- Mock de ModelPredictor ---
    predictor_inst = MagicMock()
    # load_model se llama dos veces: rf y xgb
    predictor_inst.load_model.side_effect = ["rf_model", "xgb_model"]
    mock_predictor_cls.return_value = predictor_inst

    # --- Mock de DriftDetection ---
    drift_inst = MagicMock()
    mock_drift_cls.return_value = drift_inst

    # --------- ACT ----------
    # llamamos directamente a la función main del módulo
    drift_mod.main()

    # --------- ASSERTS ----------
    # ensure_path debe llamarse al menos para processed y synthetic
    mock_ensure_path.assert_any_call(drift_mod.conf.paths.processed)
    mock_ensure_path.assert_any_call(drift_mod.conf.paths.synthetic)

    # ModelPredictor se instancia una vez con la config esperada
    mock_predictor_cls.assert_called_once()

    # load_model se llama dos veces (rf y xgb)
    assert predictor_inst.load_model.call_count == 2

    # DriftDetection se instancia y se ejecuta run()
    mock_drift_cls.assert_called_once()
    drift_inst.run.assert_called_once()
