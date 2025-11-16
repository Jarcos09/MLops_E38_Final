# tests/test_synthetic.py
import pytest
from unittest.mock import patch, MagicMock

import src.data.synthetic as synthetic_mod


@pytest.mark.unit
@patch("src.data.synthetic.paths.ensure_path")
@patch("src.data.synthetic.SyntheticDataGenerator")
@patch("src.data.synthetic.conf")
def test_preprocess_ok(mock_conf, mock_generator_cls, mock_ensure_path):
    """
    Test unitario del comando `preprocess` en src.data.synthetic.

    Verifica:
    - Llamadas a ensure_path(...)
    - Instanciación de SyntheticDataGenerator con la config correcta
    - Ejecución de run()
    """

    # ------- Config fake -------
    mock_conf.paths.synthetic = "/tmp/synth_dir"
    mock_conf.paths.models = "/tmp/models_dir"

    mock_conf.preprocessing.gmm_file = "gmm.pkl"
    mock_conf.synthetic.n_samples = 100
    mock_conf.synthetic.apply_drift = True
    mock_conf.synthetic.gamma_reweight = 0.2
    mock_conf.synthetic.lowcard_uniform_mix = 0.3
    mock_conf.synthetic.mean_shift_std = 0.5
    mock_conf.synthetic.var_scale = 1.1
    mock_conf.data.synthetic_data_file = "synthetic.csv"

    # instancia fake de SyntheticDataGenerator
    gen_instance = MagicMock()
    mock_generator_cls.return_value = gen_instance

    # ------- Llamar directamente a la función -------
    synthetic_mod.preprocess()

    # ------- Asserts -------

    # ensure_path llamado para synthetic y models
    mock_ensure_path.assert_any_call(mock_conf.paths.synthetic)
    mock_ensure_path.assert_any_call(mock_conf.paths.models)

    # Config esperada con la que se crea SyntheticDataGenerator
    expected_config = {
        "gmm_file": mock_conf.preprocessing.gmm_file,
        "n_samples": mock_conf.synthetic.n_samples,
        "apply_drift": mock_conf.synthetic.apply_drift,
        "gamma_reweight": mock_conf.synthetic.gamma_reweight,
        "lowcard_uniform_mix": mock_conf.synthetic.lowcard_uniform_mix,
        "mean_shift_std": mock_conf.synthetic.mean_shift_std,
        "var_scale": mock_conf.synthetic.var_scale,
    }

    mock_generator_cls.assert_called_once_with(
        config=expected_config,
        output_path=mock_conf.data.synthetic_data_file,
    )

    # run() del generador debe ejecutarse
    gen_instance.run.assert_called_once()
