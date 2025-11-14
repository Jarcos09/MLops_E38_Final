# tests/test_plots_model.py
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from pathlib import Path
import logging
from loguru import logger

# Forzamos backend no interactivo ANTES de importar pyplot a través del módulo
import matplotlib
matplotlib.use("Agg")  # evita necesidad de display

from src.modeling.plots_model import PlotGenerator


@pytest.mark.unit
def test_init_creates_directory(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    figures_dir = tmp_path / "figs"  # aún no existe
    assert not figures_dir.exists()

    _ = PlotGenerator(df=df, figures_path=figures_dir)

    assert figures_dir.exists()
    assert figures_dir.is_dir()


@pytest.mark.unit
def test_histogram_saves_png(tmp_path: Path, caplog):
    # Redirige loguru → logging para que caplog lo capture
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    logging_logger = logging.getLogger("plot-tests")
    logging_logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO)
    handler_id = logger.add(PropagateHandler(), level="INFO")

    try:
        df = pd.DataFrame({"x": [1,2,2,3,3,3]})
        gen = PlotGenerator(df=df, figures_path=tmp_path)
        gen.histogram(column="x", filename="hist_x.png", bins=5)

        fpath = tmp_path / "hist_x.png"
        assert fpath.exists() and fpath.stat().st_size > 0

        messages = [r.message for r in caplog.records]
        assert any("Generando histograma" in m for m in messages)
        assert any("Histograma guardado" in m for m in messages)
    finally:
        logger.remove(handler_id)

@pytest.mark.unit
@pytest.mark.parametrize("hue_col", [None, "hue"])
def test_scatter_saves_png_with_and_without_hue(tmp_path: Path, hue_col):
    # Datos mínimos: si usamos hue, añadimos columna
    data = {"x": [0, 1, 2, 3], "y": [1, 0, 1, 0]}
    if hue_col:
        data[hue_col] = ["a", "a", "b", "b"]
    df = pd.DataFrame(data)

    gen = PlotGenerator(df=df, figures_path=tmp_path)
    out_file = f"scatter_{'with_hue' if hue_col else 'no_hue'}.png"

    gen.scatter(x="x", y="y", filename=out_file, hue=hue_col)

    fpath = tmp_path / out_file
    assert fpath.exists() and fpath.stat().st_size > 0


@pytest.mark.unit
def test_histogram_raises_on_missing_column(tmp_path: Path):
    df = pd.DataFrame({"present": [1, 2, 3]})
    gen = PlotGenerator(df=df, figures_path=tmp_path)

    with pytest.raises(KeyError):
        gen.histogram(column="absent", filename="missing.png")
