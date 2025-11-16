# tests/test_paths_utils.py
from pathlib import Path
from unittest.mock import patch
import os

import pytest

from src.utils import paths as paths_mod


# ---------------------------------------------------------------------
# ensure_path
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_ensure_path_creates_directory(tmp_path):
    target_dir = tmp_path / "models" / "rf"
    assert not target_dir.exists()

    result = paths_mod.ensure_path(target_dir)

    assert isinstance(result, Path)
    assert result == target_dir
    assert target_dir.exists()
    assert target_dir.is_dir()


@pytest.mark.unit
def test_ensure_path_creates_parent_for_file(tmp_path):
    file_path = tmp_path / "models" / "rf" / "model.pkl"
    parent = file_path.parent
    assert not parent.exists()

    result = paths_mod.ensure_path(file_path)

    # Debe devolver el Path original (del archivo), pero crear el padre
    assert result == file_path
    assert parent.exists()
    assert parent.is_dir()
    # El archivo en sí no se crea
    assert not file_path.exists()


# ---------------------------------------------------------------------
# normalize_mlflow_uri
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_normalize_mlflow_uri_none():
    assert paths_mod.normalize_mlflow_uri(None) is None


@pytest.mark.unit
def test_normalize_mlflow_uri_http_passthrough():
    uri = "http://localhost:5000"
    assert paths_mod.normalize_mlflow_uri(uri) == uri


@pytest.mark.unit
def test_normalize_mlflow_uri_file_scheme_double_slash_passthrough():
    uri = "file:///abs/path/mlruns"
    assert paths_mod.normalize_mlflow_uri(uri) == uri


@pytest.mark.unit
@patch("src.utils.paths.os.path.abspath")
def test_normalize_mlflow_uri_file_short_form(mock_abspath):
    mock_abspath.return_value = "/abs/path/mlruns"
    uri = "file:./mlruns"

    normalized = paths_mod.normalize_mlflow_uri(uri)

    mock_abspath.assert_called_once_with("./mlruns")
    assert normalized == "file:///abs/path/mlruns"


# ---------------------------------------------------------------------
# build_model_registry_uri
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_build_model_registry_uri_ok():
    uri = paths_mod.build_model_registry_uri("RFRegressor", 3)
    assert uri == "models:/RFRegressor/3"


@pytest.mark.unit
def test_build_model_registry_uri_requires_version():
    with pytest.raises(ValueError):
        paths_mod.build_model_registry_uri("RFRegressor", None)


# ---------------------------------------------------------------------
# build_model_local_path
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_build_model_local_path_ok():
    path = paths_mod.build_model_local_path("RFRegressor", 2, "model.pkl")
    # Nota: la función no antepone "models/", solo <name>/<version>/<file>
    assert path == "RFRegressor/2/model.pkl"


@pytest.mark.unit
def test_build_model_local_path_requires_version():
    with pytest.raises(ValueError):
        paths_mod.build_model_local_path("RFRegressor", None, "model.pkl")


# ---------------------------------------------------------------------
# get_next_version_path
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_next_version_path_creates_first_version_when_empty(tmp_path):
    base = tmp_path / "rf_model"
    # No hay subdirectorios de versión
    result = paths_mod.get_next_version_path(base)

    assert result.parent == base
    assert result.name == "1"
    assert result.exists()
    assert result.is_dir()


@pytest.mark.unit
def test_get_next_version_path_increments_max_version(tmp_path):
    base = tmp_path / "rf_model"
    base.mkdir()

    # Crear algunas versiones existentes
    (base / "1").mkdir()
    (base / "3").mkdir()
    (base / "not_numeric").mkdir()

    result = paths_mod.get_next_version_path(base)

    # Máxima versión es 3, así que la siguiente debe ser 4
    assert result.parent == base
    assert result.name == "4"
    assert result.exists()
    assert result.is_dir()


# ---------------------------------------------------------------------
# get_latest_version_path
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_latest_version_path_creates_first_if_none(tmp_path):
    base = tmp_path / "xgb_model"
    result = paths_mod.get_latest_version_path(base)

    # Como no hay versiones, debe crear '1'
    assert result.parent == base
    assert result.name == "1"
    assert result.exists()
    assert result.is_dir()


@pytest.mark.unit
def test_get_latest_version_path_returns_max_numeric(tmp_path):
    base = tmp_path / "xgb_model"
    base.mkdir()

    (base / "1").mkdir()
    (base / "2").mkdir()
    (base / "10").mkdir()
    (base / "zzz").mkdir()  # debe ignorarse por no ser numérico

    result = paths_mod.get_latest_version_path(base)

    assert result.parent == base
    assert result.name == "10"
    assert result.exists()
    assert result.is_dir()
