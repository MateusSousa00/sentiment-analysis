import pandas as pd
import kagglehub
import os

def download_dataset():
    dataset_path = "src/data/raw/imdb_dataset.csv"

    if os.path.exists(dataset_path):
        print(f"Dataset found at {dataset_path}. Loading...")
        return pd.read_csv(dataset_path)

    print(" Dataset not found. Downloading from Kaggle...")
    downloaded_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

    os.rename(downloaded_path, dataset_path)

    print(f"Dataset saved at {dataset_path}. Loading...")
    return pd.read_csv(dataset_path)
