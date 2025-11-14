# tests/data/test_download_dataset.py

from unittest.mock import patch

import pytest

from src.data.download_dataset import DatasetDownloader


@pytest.fixture
def sample_downloader():
    return DatasetDownloader(
        dataset_id="1AbCdEfGhIjKlmNOP", 
        output_path="data/raw/dataset.csv"
    )


def test_init_sets_attributes():
    downloader = DatasetDownloader(
        dataset_id="DATASET_ID",
        output_path="data/raw/file.csv",
    )

    assert downloader.dataset_id == "DATASET_ID"
    assert downloader.output_path == "data/raw/file.csv"


@patch("src.data.download_dataset.paths.ensure_path")
def test_prepare_directory_calls_ensure_path(mock_ensure_path, sample_downloader):
    sample_downloader.prepare_directory()

    mock_ensure_path.assert_called_once_with("data/raw/dataset.csv")


@patch("src.data.download_dataset.gdown.download")
def test_download_calls_gdown_with_correct_arguments(mock_gdown_download, sample_downloader):
    sample_downloader.download()

    mock_gdown_download.assert_called_once_with(
        id="1AbCdEfGhIjKlmNOP",
        output="data/raw/dataset.csv",
        quiet=True,
    )


@patch("src.data.download_dataset.gdown.download")
@patch("src.data.download_dataset.paths.ensure_path")
def test_run_calls_prepare_directory_and_download(
    mock_ensure_path,
    mock_gdown_download,
    sample_downloader,
):
    sample_downloader.run()

    # prepare_directory
    mock_ensure_path.assert_called_once_with("data/raw/dataset.csv")

    # descarga
    mock_gdown_download.assert_called_once_with(
        id="1AbCdEfGhIjKlmNOP",
        output="data/raw/dataset.csv",
        quiet=True,
    )
