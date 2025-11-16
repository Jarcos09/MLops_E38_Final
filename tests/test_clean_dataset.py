# tests/test_clean_dataset.py

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean_dataset import DatasetCleaner


# ------------------------ Cargar datos ------------------------ #
@pytest.mark.unit
def test_init_stores_attributes():
    input_path = Path("data/raw/raw.csv")
    output_path = Path("data/processed/clean.csv")
    cleaner = DatasetCleaner(input_path=input_path,
                             output_path=output_path,
                             skew_threshold=0.5)

    assert cleaner.input_path == input_path
    assert cleaner.output_path == output_path
    assert cleaner.skew_threshold == 0.5
    assert cleaner.df is None
    # Bandera
    assert cleaner._from_dataframe is False

@pytest.mark.unit
def test_load_dataset_from_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cleaner = DatasetCleaner(input_path=df,
                             output_path=None,
                             skew_threshold=0.5)

    cleaner.load_dataset()

    assert isinstance(cleaner.df, pd.DataFrame)
    pd.testing.assert_frame_equal(cleaner.df, df)
    assert cleaner._from_dataframe is True

@pytest.mark.unit
def test_load_dataset_from_path(tmp_path: Path):
    original_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    csv_path = tmp_path / "input.csv"
    original_df.to_csv(csv_path, index=False)

    cleaner = DatasetCleaner(input_path=csv_path,
                             output_path=None,
                             skew_threshold=0.5)

    cleaner.load_dataset()

    pd.testing.assert_frame_equal(cleaner.df, original_df)

@pytest.mark.unit
def test_load_dataset_raises_when_input_none():
    cleaner = DatasetCleaner(input_path=None,
                             output_path=None,
                             skew_threshold=0.5)

    with pytest.raises(ValueError):
        cleaner.load_dataset()


# ------------------------ remplazar strings vacios ------------------------ #
@pytest.mark.unit
def test_replace_empty_strings_converts_to_nan():
    df = pd.DataFrame({
        "a": ["", "   ", "x", None],
        "b": ["1", "", "2", "   "],
    })

    cleaner = DatasetCleaner(input_path=None,
                             output_path=None,
                             skew_threshold=0.5)
    cleaner.df = df.copy()

    cleaner.replace_empty_strings()

    # Column a
    assert pd.isna(cleaner.df.loc[0, "a"])
    assert pd.isna(cleaner.df.loc[1, "a"])
    assert cleaner.df.loc[2, "a"] == "x"
    assert pd.isna(cleaner.df.loc[3, "a"])

    # Column b
    assert cleaner.df.loc[0, "b"] == "1"
    assert pd.isna(cleaner.df.loc[1, "b"])
    assert cleaner.df.loc[2, "b"] == "2"
    assert pd.isna(cleaner.df.loc[3, "b"])


# ------------------------ convertir a numerico ------------------------ #
@pytest.mark.unit
def test_convert_to_numeric_casts_and_coerces():
    df = pd.DataFrame({
        "a": ["1", "2", "x"],      # 'x' -> NaN
        "b": [1.5, "2.5", "bad"],  # "bad" -> NaN
    })

    cleaner = DatasetCleaner(input_path=None,
                             output_path=None,
                             skew_threshold=0.5)
    cleaner.df = df

    cleaner.convert_to_numeric()

    assert cleaner.df["a"].tolist()[0:2] == [1.0, 2.0]
    assert np.isnan(cleaner.df["a"].iloc[2])

    assert cleaner.df["b"].iloc[0] == 1.5
    assert cleaner.df["b"].iloc[1] == 2.5
    assert np.isnan(cleaner.df["b"].iloc[2])


# ------------------------ Imputar NaNs ------------------------ #
@pytest.mark.unit
def test_impute_missing_values_uses_mean_when_skew_small(monkeypatch):
    # mean != median para distinguir
    df = pd.DataFrame({"col": [1.0, 1.0, 100.0, np.nan]})
    cleaner = DatasetCleaner(input_path=None,
                             output_path=None,
                             skew_threshold=0.5)
    cleaner.df = df

    # Forzar kurtosis baja para usar media
    def fake_skew(series):
        return 0.0

    monkeypatch.setattr("src.data.clean_dataset.skew", fake_skew)

    cleaner.impute_missing_values()

    # media de [1, 1, 100] = 102/3 ≈ 34
    expected_mean = np.mean([1.0, 1.0, 100.0])
    assert cleaner.df["col"].isna().sum() == 0
    assert cleaner.df["col"].iloc[3] == pytest.approx(expected_mean)

@pytest.mark.unit
def test_impute_missing_values_uses_median_when_skew_large(monkeypatch):
    df = pd.DataFrame({"col": [1.0, 1.0, 100.0, np.nan]})
    cleaner = DatasetCleaner(input_path=None,
                             output_path=None,
                             skew_threshold=0.5)
    cleaner.df = df

    # kurtosis grande para usar mediana
    def fake_skew(series):
        return 10.0  # > Umbral de kurtosis

    monkeypatch.setattr("src.data.clean_dataset.skew", fake_skew)

    cleaner.impute_missing_values()

    # Mediana de [1, 1, 100] = 1
    expected_median = 1.0
    assert cleaner.df["col"].isna().sum() == 0
    assert cleaner.df["col"].iloc[3] == pytest.approx(expected_median)


# ------------------------ Guardar dataset limpio ------------------------ #
@pytest.mark.unit
def test_save_cleaned_dataset_returns_df_without_writing_when_output_none(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2, 3]})

    cleaner = DatasetCleaner(input_path=None,
                             output_path=None,  # Sin archivo
                             skew_threshold=0.5)
    cleaner.df = df

    result = cleaner.save_cleaned_dataset()

    assert result is cleaner.df  # mismo objeto
    # Si no hay archivo, no se checa

@pytest.mark.unit
def test_save_cleaned_dataset_writes_csv_to_disk(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    output_path = tmp_path / "clean.csv"

    cleaner = DatasetCleaner(input_path=None,
                             output_path=output_path,
                             skew_threshold=0.5)
    cleaner.df = df

    result = cleaner.save_cleaned_dataset()

    assert output_path.exists()
    loaded = pd.read_csv(output_path)
    # Asegura valores ya que CSV reload puede perder registros
    pd.testing.assert_frame_equal(loaded, df)
    pd.testing.assert_frame_equal(result, df)


# ------------------------ run() end-to-end ------------------------ #
@pytest.mark.integration
def test_run_with_dataframe_in_memory():
    # a tiene numerico + string vacio
    # b tiene numerico + whitespace y None
    df = pd.DataFrame({
        "a": ["1", "2", ""],
        "b": ["10", "   ", None],
    })

    cleaner = DatasetCleaner(input_path=df,
                             output_path=None,   # Trabaja en memoria
                             skew_threshold=100.0)  # Umbral grande -> mean branch
    result = cleaner.run()

    # 1) Regresa DataFrame
    assert isinstance(result, pd.DataFrame)

    # 2) Todas las columnas deben ser numericas
    assert all(np.issubdtype(dtype, np.number) for dtype in result.dtypes)

    # 3) No NaNs despues de imputacion de datos
    assert result.isna().sum().sum() == 0
