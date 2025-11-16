# tests/test_preprocess_data.py
# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.preprocess_data import DataPreprocessor


# ---------------------------------------------------------------------
# Utilidad: configuración mínima común
# ---------------------------------------------------------------------
def base_config(tmp_path: Path):
    """Devuelve una config válida y compacta para pruebas."""
    return {
        "target_columns": ["y"],
        "feature_columns": ["f1", "f2"],
        "allow_missing_columns": False,
        "encoding": {
            "drop": "first",
            "sparse_output": False,
            "handle_unknown": "ignore",
        },
        "test_size": 0.4,
        "random_state": 0,
        "target_transform": "yeo-johnson",
        "preprocessor_file": str(tmp_path / "preprocessor.pkl"),

        #  Claves usadas por generate_gmm()
        "gmm_n_components": 2,
        "gmm_random_state": 0,
        "gmm_file": str(tmp_path / "gmm.json"),
    }




# ---------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_load_data_reads_csv(tmp_path: Path):
    df = pd.DataFrame({"f1": ["a", "b", "a"], "f2": ["x", "y", "y"], "y": [1.0, 2.0, 3.0]})
    csv = tmp_path / "clean.csv"
    df.to_csv(csv, index=False)

    cfg = base_config(tmp_path)
    pre = DataPreprocessor(input_path=csv, output_paths={}, config=cfg)
    pre.load_data()

    pd.testing.assert_frame_equal(pre.df, df)


# ---------------------------------------------------------------------
# separate_variables
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_separate_variables_ok(tmp_path: Path):
    cfg = base_config(tmp_path)
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths={}, config=cfg)
    pre.df = pd.DataFrame({"f1": ["a", "b", "a"], "f2": ["x", "y", "y"], "y": [1, 0, 1], "extra": [9, 9, 9]})

    pre.separate_variables()

    assert list(pre.X.columns) == ["f1", "f2"]
    assert list(pre.y.columns) == ["y"]
    assert pre.X.shape == (3, 2)
    assert pre.y.shape == (3, 1)


@pytest.mark.unit
def test_separate_variables_missing_forbidden_raises(tmp_path: Path):
    cfg = base_config(tmp_path)
    cfg["allow_missing_columns"] = False
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths={}, config=cfg)
    # Falta f2
    pre.df = pd.DataFrame({"f1": ["a", "b"], "y": [1, 0]})

    with pytest.raises(ValueError):
        pre.separate_variables()


@pytest.mark.unit
def test_separate_variables_missing_allowed_fills_nan(tmp_path: Path):
    cfg = base_config(tmp_path)
    cfg["allow_missing_columns"] = True
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths={}, config=cfg)
    # Falta f2
    pre.df = pd.DataFrame({"f1": ["a", "b"], "y": [1, 0]})

    pre.separate_variables()

    # Debe crear f2 con NaN y mantener orden de columnas
    assert list(pre.X.columns) == ["f1", "f2"]
    assert pre.X["f2"].isna().all()
    assert list(pre.y.columns) == ["y"]


# ---------------------------------------------------------------------
# encode_features
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_encode_features_builds_pipeline(tmp_path: Path):
    cfg = base_config(tmp_path)
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths={}, config=cfg)
    pre.X = pd.DataFrame({"f1": ["a", "b", "a"], "f2": ["x", "y", "y"]})

    pre.encode_features()

    assert pre.pipeline is not None
    # El pipeline debe tener un ColumnTransformer con nuestro encoder
    assert "preprocessor" in pre.pipeline.named_steps
    transformer = pre.pipeline.named_steps["preprocessor"]
    assert transformer is not None


# ---------------------------------------------------------------------
# split_and_transform
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_split_and_transform_shapes_and_features(tmp_path: Path):
    cfg = base_config(tmp_path)
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths={}, config=cfg)
    pre.X = pd.DataFrame({"f1": ["a", "b", "a", "b", "a"], "f2": ["x", "y", "y", "x", "x"]})
    pre.y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})

    pre.encode_features()
    Xtr, Xte, ytr, yte, idx_tr, idx_te = pre.split_and_transform()

    # Shapes coherentes con test_size=0.4 para 5 filas → 3 train, 2 test
    assert Xtr.shape[0] == 3 and Xte.shape[0] == 2
    assert ytr.shape == (3, 1) and yte.shape == (2, 1)
    # Nombres de features deben coincidir con columnas del transformador
    assert len(pre.feature_names) == Xtr.shape[1]


# ---------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_save_outputs_write_csvs(tmp_path: Path):
    cfg = base_config(tmp_path)
    out = {
        "X_TRAIN": str(tmp_path / "X_train.csv"),
        "X_TEST": str(tmp_path / "X_test.csv"),
        "Y_TRAIN": str(tmp_path / "y_train.csv"),
        "Y_TEST": str(tmp_path / "y_test.csv"),
    }
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths=out, config=cfg)
    pre.feature_names = np.array(["f1_b", "f2_y"])

    # Datos ficticios
    Xtr = np.array([[1, 0], [0, 1], [1, 0]])
    Xte = np.array([[0, 1]])
    ytr = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    yte = pd.DataFrame({"y": [4.0]})
    idx_tr = pd.RangeIndex(3)
    idx_te = pd.RangeIndex(1)

    pre.save_outputs(Xtr, Xte, ytr, yte, idx_tr, idx_te)

    # Archivos deben existir y tener el número correcto de filas
    for k, p in out.items():
        assert Path(p).exists()
    assert pd.read_csv(out["X_TRAIN"]).shape == (3, 2)
    assert pd.read_csv(out["Y_TEST"]).shape == (1, 1)


# ---------------------------------------------------------------------
# save_preprocessor
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_save_preprocessor_raises_if_no_pipeline(tmp_path: Path):
    cfg = base_config(tmp_path)
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths={}, config=cfg)
    pre.pipeline = None
    with pytest.raises(RuntimeError):
        pre.save_preprocessor(dest=tmp_path / "pp.pkl")


@pytest.mark.unit
def test_save_preprocessor_dump_ok(tmp_path: Path):
    cfg = base_config(tmp_path)
    pre = DataPreprocessor(input_path=tmp_path / "x.csv", output_paths={}, config=cfg)
    # Construimos un pipeline real para que joblib pueda serializar
    pre.X = pd.DataFrame({"f1": ["a", "b"], "f2": ["x", "y"]})
    pre.encode_features()

    dest = tmp_path / "pp.pkl"
    result_path = pre.save_preprocessor(dest)

    assert Path(result_path).exists()
    assert Path(result_path) == dest


# ---------------------------------------------------------------------
# run (end-to-end con archivos temporales)
# ---------------------------------------------------------------------
@pytest.mark.integration
def test_run_end_to_end(tmp_path: Path, monkeypatch):
    # CSV de entrada
    df = pd.DataFrame(
        {"f1": ["a", "b", "a", "b", "a"], "f2": ["x", "y", "y", "x", "x"], "y": [1, 2, 3, 4, 5]}
    )
    clean_csv = tmp_path / "clean.csv"
    df.to_csv(clean_csv, index=False)

    out = {
        "X_TRAIN": str(tmp_path / "X_train.csv"),
        "X_TEST": str(tmp_path / "X_test.csv"),
        "Y_TRAIN": str(tmp_path / "y_train.csv"),
        "Y_TEST": str(tmp_path / "y_test.csv"),
    }
    cfg = base_config(tmp_path)

    pre = DataPreprocessor(input_path=clean_csv, output_paths=out, config=cfg)

    # No queremos ruido si save_preprocessor falla por cualquier motivo
    # pero como config tiene preprocessor_file válido, debería guardarse bien.
    pre.run()

    # Deben existir los 4 archivos de salida y el preprocessor
    for p in out.values():
        assert Path(p).exists()
    assert Path(cfg["preprocessor_file"]).exists()
