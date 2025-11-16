import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data.synthetic_generator import SyntheticDataGenerator


# ============================================================
#                 FIXTURE CONFIG BÁSICA
# ============================================================

@pytest.fixture
def base_config(tmp_path):
    return {
        "gmm_file": str(tmp_path / "gmm.pkl"),
        "n_samples": 10,
        "apply_drift": False,
        "gamma_reweight": 0.2,
        "lowcard_uniform_mix": 0.3,
        "mean_shift_std": 0.2,
        "var_scale": 1.1,
    }


# ============================================================
#                 TEST load_gmm()
# ============================================================

@pytest.mark.unit
@patch("src.data.synthetic_generator.joblib.load")
def test_load_gmm_ok(mock_joblib_load, base_config):
    fake_gmm = MagicMock()
    payload = {"model": fake_gmm, "feature_names": ["a", "b"]}
    mock_joblib_load.return_value = payload

    gen = SyntheticDataGenerator(base_config, "out.csv")
    gen.load_gmm()

    mock_joblib_load.assert_called_once_with(base_config["gmm_file"])
    assert gen.gmm is fake_gmm
    assert gen.feature_names == ["a", "b"]


@pytest.mark.unit
@patch("src.data.synthetic_generator.joblib.load")
def test_load_gmm_missing_model_key(mock_joblib_load, base_config):
    mock_joblib_load.return_value = {"wrong": 123}

    gen = SyntheticDataGenerator(base_config, "out.csv")

    with pytest.raises(KeyError):
        gen.load_gmm()

@pytest.mark.unit
def test_load_gmm_missing_file_raises(base_config):
    gen = SyntheticDataGenerator(base_config, "out.csv")

    with pytest.raises(Exception):
        gen.load_gmm()


# ============================================================
#                 TEST generate()
# ============================================================

@pytest.mark.unit
def test_generate_ok_with_feature_names():
    fake_gmm = MagicMock()
    fake_gmm.sample.return_value = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        None,
    )

    cfg = {"n_samples": 2}
    gen = SyntheticDataGenerator(cfg, "out.csv")
    gen.gmm = fake_gmm
    gen.feature_names = ["x1", "x2"]

    df = gen.generate()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x1", "x2"]
    assert len(df) == 2


@pytest.mark.unit
def test_generate_without_feature_names():
    fake_gmm = MagicMock()
    fake_gmm.sample.return_value = (
        np.array([[1.0], [2.0]]),
        None,
    )

    cfg = {"n_samples": 2}
    gen = SyntheticDataGenerator(cfg, "out.csv")
    gen.gmm = fake_gmm
    gen.feature_names = None

    df = gen.generate()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 1)  # columnas numéricas


@pytest.mark.unit
def test_generate_raises_if_no_gmm():
    gen = SyntheticDataGenerator({"n_samples": 5}, "out.csv")

    with pytest.raises(ValueError):
        gen.generate()


# ============================================================
#           TEST _classify_columns()
# ============================================================

@pytest.mark.unit
def test_classify_columns():
    # Usamos 20 filas para poder tener columna continua (>15 valores únicos)
    n = 20

    df = pd.DataFrame(
        {
            # binaria 0/1
            "bin1": ([0, 1] * (n // 2))[:n],

            # lowcard numérica (3–15 categorías)
            "low": ([1, 2, 3, 1, 2] * (n // 5))[:n],

            # continua con muchos valores distintos (>15)
            "cont": np.linspace(0.0, 1.0, n),
        }
    )

    gen = SyntheticDataGenerator({}, "out.csv")
    binary, low, cont = gen._classify_columns(df)

    assert "bin1" in binary
    assert "low" in low
    assert "cont" in cont


# ============================================================
#           TEST _apply_drift()
# ============================================================

@pytest.mark.unit
def test_apply_drift_changes_values(base_config):
    input_df = pd.DataFrame(
        {
            "bin1": [0, 1, 0, 1],
            "low": [1, 2, 3, 1],
            "cont": [10.0, 11.0, 9.0, 10.5],
        }
    )

    gen = SyntheticDataGenerator(base_config, "out.csv")
    df_out = gen._apply_drift(input_df)

    assert not df_out.equals(input_df)
    assert len(df_out) == len(input_df)


# ============================================================
#           TEST run() COMPLETO CON MOCKS
# ============================================================

@pytest.mark.unit
@patch("src.data.synthetic_generator.SyntheticDataGenerator.save")
@patch("src.data.synthetic_generator.SyntheticDataGenerator.generate")
@patch("src.data.synthetic_generator.SyntheticDataGenerator.load_gmm")
def test_run_executes_all(mock_load, mock_gen, mock_save, base_config):
    gen = SyntheticDataGenerator(base_config, "out.csv")

    df_fake = pd.DataFrame({"a": [1, 2]})
    mock_gen.return_value = df_fake

    gen.run()

    mock_load.assert_called_once()
    mock_gen.assert_called_once()
    mock_save.assert_called_once_with(df_fake)
