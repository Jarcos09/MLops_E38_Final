# tests/test_train_model.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.modeling.train_model import ModelTrainer


# ------------------------- util: config base -------------------------- #
def base_cfg(tmp_path: Path):
    return {
        "mlflow_tracking_uri": "file:///tmp/mlruns",
        "random_state": 0,
        "best_metric": "rmse",
        # nombres / rutas para registro y serialización
        "rf_experiment_name": "exp_rf",
        "rf_registry_model_name": "RFRegressor",
        "rf_model_path": tmp_path / "models_rf",
        "rf_model_file": "rf.pkl",
        "xgb_experiment_name": "exp_xgb",
        "xgb_registry_model_name": "XGBRegressor",
        "xgb_model_path": tmp_path / "models_xgb",
        "xgb_model_file": "xgb.pkl",
        # grids minimales para que las pruebas sean rápidas
        "rf_param_grid": {
            # Los nombres deben llevar el prefijo estimator__ como espera el código
            "estimator__n_estimators": [5],
            "estimator__max_depth": [3],
            "estimator__min_samples_split": [2],
        },
        "xgb_param_grid": {
            "learning_rate": [0.1],
            "max_depth": [2],
            "n_estimators": [5],
            "subsample": [1.0],
        },
    }


# ---------------------- fixture de datos pequeños --------------------- #
@pytest.fixture
def tiny_data():
    # dataset pequeño multi-salida (2 targets)
    X = pd.DataFrame({"f1": [0, 1, 2, 3, 4, 5], "f2": [1, 1, 2, 3, 5, 8]})
    y = pd.DataFrame({"t0": [1.0, 1.1, 1.9, 3.0, 4.1, 5.1],
                      "t1": [0.5, 0.7, 1.0, 1.6, 2.0, 2.4]})
    # split simple 4/2
    X_train, X_test = X.iloc[:4], X.iloc[4:]
    y_train, y_test = y.iloc[:4], y.iloc[4:]
    return X_train, X_test, y_train, y_test


# ------------------ log_metrics: cálculo y logging -------------------- #
class _DummyRun:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False


@pytest.mark.unit
@patch("src.modeling.train_model.mlflow")
def test_log_metrics_computes_and_logs(mock_mlflow, tiny_data, tmp_path):
    Xtr, Xte, ytr, yte = tiny_data
    cfg = base_cfg(tmp_path)
    mt = ModelTrainer(Xtr, Xte, ytr, yte, cfg)

    # predicciones "cercanas" (2 columnas)
    y_pred = np.column_stack([
        yte["t0"].to_numpy() + np.array([0.1, -0.1]),
        yte["t1"].to_numpy() + np.array([0.0, 0.05]),
    ])

    metrics = mt.log_metrics(y_pred)

    # cheques básicos de contenido
    assert "rmse_t0" in metrics and "r2_t0" in metrics
    assert "rmse_t1" in metrics and "r2_t1" in metrics
    assert "avg_rmse" in metrics and "avg_r2" in metrics

    # debe loguear a mlflow
    mock_mlflow.log_metrics.assert_called_once()
    logged = mock_mlflow.log_metrics.call_args.args[0]
    assert set(metrics.keys()) <= set(logged.keys())


# -------------------- log_model: serializa y registra ----------------- #
@pytest.mark.unit
@patch("src.modeling.train_model.mlflow")
@patch("src.modeling.train_model.paths")
def test_log_model_serializes_and_logs(mock_paths, mock_mlflow, tiny_data, tmp_path):
    Xtr, Xte, ytr, yte = tiny_data
    cfg = base_cfg(tmp_path)
    mt = ModelTrainer(Xtr, Xte, ytr, yte, cfg)

    # mock get_next_version_path para que cree un directorio real
    dest_dir = tmp_path / "models_rf" / "rev_0001"
    def _next_version(p):
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir
    mock_paths.get_next_version_path.side_effect = _next_version

    # modelo multioutput pequeño: entrenamos rápidamente para poder picklearlo
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    mdl = MultiOutputRegressor(RandomForestRegressor(n_estimators=5, random_state=0))
    mdl.fit(Xtr, ytr)

    # sin pasar params: el código intentará extraer get_params() del estimador base
    mt.log_model(
        model=mdl,
        model_name=cfg["rf_registry_model_name"],
        model_path=cfg["rf_model_path"],
        model_file=cfg["rf_model_file"],
        params=None,
    )

    # archivo debe existir
    expected_file = dest_dir / cfg["rf_model_file"]
    assert expected_file.exists() and expected_file.stat().st_size > 0

    # mlflow debe registrar modelo, artifact y tags
    assert mock_mlflow.sklearn.log_model.called
    assert mock_mlflow.log_artifact.called
    assert mock_mlflow.set_tag.call_count >= 3  # model_stage, model_type, framework, ...


# --------------- train_random_forest: integración ligera -------------- #
@pytest.mark.integration
@patch("src.modeling.train_model.mlflow")
@patch("src.modeling.train_model.paths")
def test_train_random_forest_returns_best_model_and_metrics(
    mock_paths, mock_mlflow, tiny_data, tmp_path
):
    Xtr, Xte, ytr, yte = tiny_data
    cfg = base_cfg(tmp_path)

    # start_run debe comportarse como context manager
    mock_mlflow.start_run.return_value = _DummyRun()

    # Para evitar escribir fuera del tmp, devolvemos una carpeta real
    rf_dest = tmp_path / "rf_rev"
    def _next_version(p):
        rf_dest.mkdir(parents=True, exist_ok=True)
        return rf_dest
    mock_paths.get_next_version_path.side_effect = _next_version

    mt = ModelTrainer(Xtr, Xte, ytr, yte, cfg)
    best_model, final_metrics = mt.train_random_forest()

    # regresó modelo y métricas
    from sklearn.multioutput import MultiOutputRegressor
    assert isinstance(best_model, MultiOutputRegressor)
    assert "avg_rmse" in final_metrics and "avg_r2" in final_metrics

    # set_experiment y varias llamadas de logging
    mock_mlflow.set_experiment.assert_called_once_with(cfg["rf_experiment_name"])
    assert mock_mlflow.log_params.called
    assert mock_mlflow.log_metrics.called
    assert mock_mlflow.sklearn.log_model.called

    # el mejor modelo queda en la instancia
    assert hasattr(mt, "best_rf_model") and mt.best_rf_model is best_model

    # se generó archivo del modelo
    saved_file = rf_dest / cfg["rf_model_file"]
    assert saved_file.exists()


# -------------------- (opcional) train_xgboost smoke test -------------------- #
#
# @pytest.mark.unit
# @patch("src.modeling.train_model.mlflow")
# @patch("src.modeling.train_model.paths")
# def test_train_xgboost_smoke(mock_paths, mock_mlflow, tiny_data, tmp_path):
#     Xtr, Xte, ytr, yte = tiny_data
#     cfg = base_cfg(tmp_path)
#     mock_mlflow.start_run.return_value = _DummyRun()
#     xgb_dest = tmp_path / "xgb_rev"
#     mock_paths.get_next_version_path.side_effect = lambda p: xgb_dest.mkdir(parents=True, exist_ok=True) or xgb_dest
#     mt = ModelTrainer(Xtr, Xte, ytr, yte, cfg)
#     model, metrics = mt.train_xgboost()
#     assert "avg_rmse" in metrics
