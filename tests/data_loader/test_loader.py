import os
import pytest
import pandas as pd
from unittest import mock
from src.data_loader import download_dataset

@pytest.fixture
def mock_csv_file(tmp_path):
    """Create a fake CSV file"""
    dataset_path = tmp_path / "imdb_dataset.csv"
    df = pd.DataFrame({"review": ["Good movie!", "Bad movie!"], "sentiment": [1, 0]})
    df.to_csv(dataset_path, index=False)
    return dataset_path

def test_download_dataset_file_exists(mock_csv_file, monkeypatch):
    """Test if dataset loads correctly when the file exists"""
    
    dataset_path = "src/data/raw/imdb_dataset.csv"
    monkeypatch.setattr("src.data_loader.os.path.exists", lambda x: x == dataset_path)

    with mock.patch("pandas.read_csv", return_value=pd.read_csv(mock_csv_file)) as mock_read:
        df = download_dataset()
        mock_read.assert_called_once_with(dataset_path)
        assert not df.empty, "Dataset should be loaded when file exists"

@mock.patch("src.data_loader.kagglehub.dataset_download")
@mock.patch("src.data_loader.os.rename")
def test_download_dataset_not_found(mock_rename, mock_download, monkeypatch):
    """Test dataset download logic when file does not exist"""
    monkeypatch.setattr("src.data_loader.os.path.exists", lambda x: False)

    fake_download_path = "temp_downloaded_file.csv"
    mock_download.return_value = fake_download_path

    df = download_dataset()

    mock_download.assert_called_once_with("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    mock_rename.assert_called_once_with(fake_download_path, "src/data/raw/imdb_dataset.csv")

    assert not df.empty, "Dataset should be loaded after downloading"
