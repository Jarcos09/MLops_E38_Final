# tests/test_drift_detection.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.modeling.drift_detection import DriftDetection


# ---------------------------------------------------------------------
# Dummmy model para las pruebas
# ---------------------------------------------------------------------
class DummyModel:
    """Modelo muy simple: predice siempre 0 (o la media) para mantener estable el test."""

    def __init__(self, strategy: str = "zeros"):
        self.strategy = strategy

    def predict(self, X: pd.DataFrame):
        n = len(X)
        if self.strategy == "zeros":
            return np.zeros(n)
        elif self.strategy == "ones":
            return np.ones(n)
        else:
            # media creciente para dar algo de variación
            return np.linspace(0, 1, n)


# ---------------------------------------------------------------------
# load_dataset: pruebas UNITARIAS
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_load_dataset_from_csv_ok(tmp_path: Path):
    """load_dataset debe cargar X, y y el sintético desde CSV, recortar y alinear columnas."""

    # --- Crear CSVs base ---
    X_val = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
        }
    )
    y_val = pd.DataFrame({"target": [0.0, 1.0, 0.0, 1.0]})

    X_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    synth_path = tmp_path / "X_synth.csv"

    X_val.to_csv(X_path, index=False)
    y_val.to_csv(y_path, index=False)

    # sintético con mismas columnas + alguna extra
    X_synth = X_val.copy()
    X_synth["extra"] = [10, 20, 30, 40]
    X_synth.to_csv(synth_path, index=False)

    det = DriftDetection(
        X_path=X_path,
        y_path=y_path,
        synthetic_data_source=synth_path,
        model=DummyModel(),
    )

    det.load_dataset()

    # y_val_for_drift debe existir
    assert hasattr(det, "y_val_for_drift")

    # Misma cantidad de filas en todos los datasets
    n = len(X_val)
    assert det.X_val.shape[0] == n
    assert det.X_drift.shape[0] == n
    assert len(det.y_val_for_drift) == n

    # Columnas: X_drift debe estar alineado con X_val (sin "extra")
    assert list(det.X_val.columns) == ["num", "cat"]
    assert list(det.X_drift.columns) == ["num", "cat"]


@pytest.mark.unit
def test_load_dataset_uses_dataframe_source(tmp_path: Path):
    """Cuando synthetic_data_source es un DataFrame, debe usarse directamente."""

    X_val = pd.DataFrame({"x": [1, 2, 3], "y": [0.5, 0.7, 0.9]})
    y_val = pd.DataFrame({"target": [0.0, 1.0, 0.0]})

    X_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"

    X_val.to_csv(X_path, index=False)
    y_val.to_csv(y_path, index=False)

    # sintético en memoria
    X_synth = pd.DataFrame({"x": [10, 20, 30], "y": [1.5, 1.7, 1.9]})

    det = DriftDetection(
        X_path=X_path,
        y_path=y_path,
        synthetic_data_source=X_synth,
        model=DummyModel(),
    )

    det.load_dataset()

    # Debe haberse copiado el DataFrame (no la misma referencia)
    assert isinstance(det.X_drift, pd.DataFrame)
    assert not det.X_drift is X_synth
    assert det.X_drift.equals(X_synth.reset_index(drop=True))


@pytest.mark.unit
def test_load_dataset_raises_if_missing_columns_in_drift(tmp_path: Path):
    """Debe lanzar ValueError si X_drift no contiene todas las columnas de X_val."""

    X_val = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y_val = pd.DataFrame({"target": [0, 1, 0]})

    X_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    synth_path = tmp_path / "X_synth.csv"

    X_val.to_csv(X_path, index=False)
    y_val.to_csv(y_path, index=False)

    # sintético sin la columna 'b'
    X_synth = pd.DataFrame({"a": [10, 20, 30]})
    X_synth.to_csv(synth_path, index=False)

    det = DriftDetection(
        X_path=X_path,
        y_path=y_path,
        synthetic_data_source=synth_path,
        model=DummyModel(),
    )

    with pytest.raises(ValueError, match="X_drift NO tiene columnas necesarias"):
        det.load_dataset()


# ---------------------------------------------------------------------
# _classify_columns_and_select_drift: prueba de INTEGRACIÓN
# ---------------------------------------------------------------------
@pytest.mark.integration
def test_drift_report_contains_expected_structure_and_flags(tmp_path: Path):
    """
    Prueba de integración: se cargan datos reales, se ejecuta toda la lógica
    de _classify_columns_and_select_drift y se verifica que el reporte
    tenga las columnas clave y detecte drift en al menos una feature.
    """

    np.random.seed(0)

    # --- Dataset base (val) ---
    n = 200
    X_val = pd.DataFrame(
        {
            # numérica con media 0
            "num_stable": np.random.normal(0, 1, size=n),
            # numérica con drift fuerte (sintético tendrá media muy distinta)
            "num_drift": np.random.normal(0, 1, size=n),
            # categórica estable
            "cat_stable": np.random.choice(["a", "b"], size=n),
            # categórica con drift
            "cat_drift": np.random.choice(["x", "y"], size=n),
        }
    )
    # target simple (no nos interesa el valor exacto aquí)
    y_val = pd.DataFrame({"target": np.random.randint(0, 2, size=n)})

    X_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    X_val.to_csv(X_path, index=False)
    y_val.to_csv(y_path, index=False)

    # --- Dataset sintético con drift marcado ---
    X_synth = pd.DataFrame(
        {
            "num_stable": np.random.normal(0, 1, size=n),          # similar
            "num_drift": np.random.normal(10, 1, size=n),          # media muy distinta
            "cat_stable": np.random.choice(["a", "b"], size=n),    # similar
            "cat_drift": np.random.choice(["x", "z"], size=n),     # nueva categoría 'z'
        }
    )

    det = DriftDetection(
        X_path=X_path,
        y_path=y_path,
        synthetic_data_source=X_synth,
        model=DummyModel(strategy="zeros"),
    )

    det.load_dataset()
    drift_report = det._classify_columns_and_select_drift()

    # Estructura mínima esperada
    for col in ["feature", "type", "test", "p_value", "drift_detected", "severity"]:
        assert col in drift_report.columns

    # Debe existir al menos una columna con drift_detected = True
    assert drift_report["drift_detected"].any()

    # La columna num_drift muy probablemente debe aparecer con drift detectado
    num_drift_row = drift_report[drift_report["feature"] == "num_drift"].iloc[0]
    assert num_drift_row["drift_detected"]

