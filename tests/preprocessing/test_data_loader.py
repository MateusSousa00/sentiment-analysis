import pytest
import pandas as pd
from src.data_loader import download_dataset

def test_load_dataset():
    dataset_path = download_dataset()
    assert dataset_path is not None, "Dataset path should not be None"

    df = pd.read_csv(dataset_path)
    assert not df.empty, "Dataset should not be empty"

if __name__ == "__main__":
    pytest.main()
