import os
import kagglehub
import shutil

RAW_DATA_DIR = "src/data/raw"
EXPECTED_CSV_NAME = "imdb_dataset.csv"

def download_dataset():
    """Downloads dataset from Kaggle and ensures it's saved in the correct location."""
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    expected_final_path = os.path.join(RAW_DATA_DIR, EXPECTED_CSV_NAME)

    if os.path.exists(expected_final_path):
        print(f"Dataset already exists at {expected_final_path}.")
        return expected_final_path


    print("📥 Downloading dataset from KaggleHub...")
    
    try:
        extracted_dir = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

        # Find the actual CSV file in the extracted directory
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                if file.endswith(".csv"):
                    full_csv_path = os.path.join(root, file)
                    
                    # Move and rename to standard path
                    shutil.move(full_csv_path, expected_final_path)
                    print(f"Dataset saved as {expected_final_path}")
                    return expected_final_path

        print("No CSV file found in the downloaded dataset.")
        return None

    except Exception as e:
        print(f"Dataset download failed: {e}")
        return None
