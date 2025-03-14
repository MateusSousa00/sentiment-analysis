import os
import tempfile
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

    result = download_dataset()
    assert result == dataset_path, "Expected dataset path to be returned when file exists"

@mock.patch("src.data_loader.kagglehub.dataset_download")
@mock.patch("shutil.move")
def test_download_dataset_not_found(mock_move, mock_download, monkeypatch):
    """Test dataset download logic when file does not exist"""

    monkeypatch.setattr("src.data_loader.os.path.exists", lambda x: False)

    fake_download_dir = tempfile.mkdtemp()
    fake_csv_file = os.path.join(fake_download_dir, "imdb_dataset.csv")

    # Simulate the file existing in the extracted directory
    with open(fake_csv_file, "w") as f:
        f.write("dummy data")

    mock_download.return_value = fake_download_dir

    dataset_path = download_dataset()

    mock_move.assert_called_once_with(fake_csv_file, "src/data/raw/imdb_dataset.csv")
    assert dataset_path == "src/data/raw/imdb_dataset.csv", "Expected dataset path after download"
