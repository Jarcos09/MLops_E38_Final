# tests/test_mlflow_client.py
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.utils.mlflow_client import MLFlowClient


# -------------------------------------------------------------------
# __init__
# -------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_client.MlflowClient")
@patch("src.utils.mlflow_client.mlflow.set_tracking_uri")
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="http://mlflow-server")
def test_init_sets_tracking_uri_and_client(mock_norm, mock_set_uri, mock_mlflow_client_cls):
    inst = MLFlowClient(tracking_uri="http://mlflow-server")

    mock_norm.assert_called_once_with("http://mlflow-server")
    mock_set_uri.assert_called_once_with("http://mlflow-server")
    mock_mlflow_client_cls.assert_called_once_with(tracking_uri="http://mlflow-server")

    assert inst.tracking_uri == "http://mlflow-server"
    assert inst.client is mock_mlflow_client_cls.return_value


# -------------------------------------------------------------------
# check_remote_available
# -------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_client.requests.get")
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="http://mlflow-server")
def test_check_remote_available_http_ok(mock_norm, mock_get):
    inst = MLFlowClient(tracking_uri="http://mlflow-server")

    resp = MagicMock()
    mock_get.return_value = resp

    inst.check_remote_available()

    mock_get.assert_called_once_with(
        "http://mlflow-server/api/2.0/mlflow/registered-models/search",
        json={},
        timeout=inst.http_timeout,
    )
    resp.raise_for_status.assert_called_once()


@pytest.mark.unit
@patch("src.utils.mlflow_client.requests.get")
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="file:/tmp/mlruns")
def test_check_remote_available_non_http_does_nothing(mock_norm, mock_get):
    inst = MLFlowClient(tracking_uri="file:/tmp/mlruns")

    # No debe llamar a requests.get para URIs no HTTP
    inst.check_remote_available()
    mock_get.assert_not_called()


@pytest.mark.unit
@patch("src.utils.mlflow_client.requests.get", side_effect=Exception("boom"))
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="http://mlflow-server")
def test_check_remote_available_http_raises(mock_norm, mock_get):
    inst = MLFlowClient(tracking_uri="http://mlflow-server")

    with pytest.raises(Exception):
        inst.check_remote_available()


# -------------------------------------------------------------------
# get_model_version
# -------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_client.MlflowClient")
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="http://mlflow-server")
def test_get_model_version_ok(mock_norm, mock_mlflow_client_cls):
    client_inst = mock_mlflow_client_cls.return_value
    mv = SimpleNamespace(
        version="3",
        current_stage="Production",
        source="s3://bucket/model",
        run_id="run-123",
    )
    client_inst.get_model_version.return_value = mv

    inst = MLFlowClient(tracking_uri="http://mlflow-server")

    result = inst.get_model_version("my_model", "3")

    client_inst.get_model_version.assert_called_once_with(name="my_model", version="3")
    assert result == {
        "version": "3",
        "stage": "Production",
        "source": "s3://bucket/model",
        "run_id": "run-123",
    }


@pytest.mark.unit
@patch("src.utils.mlflow_client.MlflowClient")
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="http://mlflow-server")
def test_get_model_version_error_raises_runtimeerror(mock_norm, mock_mlflow_client_cls):
    client_inst = mock_mlflow_client_cls.return_value
    client_inst.get_model_version.side_effect = Exception("not found")

    inst = MLFlowClient(tracking_uri="http://mlflow-server")

    with pytest.raises(RuntimeError) as exc:
        inst.get_model_version("my_model", "99")

    assert "No se pudo obtener la versión 99 del modelo 'my_model'" in str(exc.value)


# -------------------------------------------------------------------
# get_latest_version
# -------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value=None)
def test_get_latest_version_picks_max_numeric_version(mock_norm):
    inst = MLFlowClient(tracking_uri=None)

    mv1 = SimpleNamespace(version="1", current_stage="Staging", source="s1", run_id="r1")
    mv2 = SimpleNamespace(version="2", current_stage="Production", source="s2", run_id="r2")
    mv10 = SimpleNamespace(version="10", current_stage="Production", source="s10", run_id="r10")

    # search_model_versions se llama 2 veces dentro de get_latest_version
    inst.search_model_versions = MagicMock(return_value=[mv1, mv2, mv10])

    result = inst.get_latest_version("my_model")

    assert result == {
        "version": "10",
        "stage": "Production",
        "source": "s10",
        "run_id": "r10",
    }


@pytest.mark.unit
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value=None)
def test_get_latest_version_returns_none_when_no_versions(mock_norm):
    inst = MLFlowClient(tracking_uri=None)
    inst.search_model_versions = MagicMock(return_value=[])

    assert inst.get_latest_version("empty_model") is None


# -------------------------------------------------------------------
# render_models_table
# -------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="http://mlflow-server")
def test_render_models_table_with_models(mock_norm):
    inst = MLFlowClient(tracking_uri="http://mlflow-server")

    rm1 = SimpleNamespace(name="modelA")
    inst.list_registered_models = MagicMock(return_value=[rm1])

    mv1 = SimpleNamespace(version="1")
    mv2 = SimpleNamespace(version="2")
    inst.search_model_versions = MagicMock(return_value=[mv1, mv2])

    table = inst.render_models_table()

    # Debe incluir la URI, el nombre del modelo y las versiones
    assert "MLflow Tracking URI: http://mlflow-server" in table
    assert "modelA" in table
    assert "1, 2" in table or "2, 1" in table


@pytest.mark.unit
@patch("src.utils.mlflow_client.paths.normalize_mlflow_uri", return_value="http://mlflow-server")
def test_render_models_table_no_models(mock_norm):
    inst = MLFlowClient(tracking_uri="http://mlflow-server")
    inst.list_registered_models = MagicMock(return_value=[])

    table = inst.render_models_table()
    assert table.strip() == "No hay modelos registrados en MLflow."
