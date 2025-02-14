import pytest
from src.data_loader import download_dataset

def test_load_dataset():
    df = download_dataset()
    assert df is not None
    assert not df.empty
    assert "review" in df.columns
    assert "sentiment" in df.columns

if __name__ == "__main__":
    pytest.main()
