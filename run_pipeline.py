import os
from src.data_loader import download_dataset
from src.preprocessing.prepare_dataset import preprocess_dataset
from src.training.train_baseline_model import train_baseline_model

# Ensure necessary directories exist
os.makedirs("src/data/raw", exist_ok=True)
os.makedirs("src/data/processed", exist_ok=True)

def main():
    print("Downloading dataset...")
    dataset_path = download_dataset()
    if not dataset_path:
        print("Failed to download dataset. Exiting.")
        exit(1)

    print("Preprocessing dataset...")
    preprocess_dataset()

    print("Training baseline model...")
    train_baseline_model()

    print("Sentiment Analysis Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
