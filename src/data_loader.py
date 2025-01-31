import kagglehub
import os
import pandas as pd

def download_dataset():
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    return os.path.join(path, csv_file)

if __name__ == "__main__":
    dataset_path = download_dataset()
    df = pd.read_csv(dataset_path)
    df.to_csv("data/raw/imdb_dataset.csv", index=False)