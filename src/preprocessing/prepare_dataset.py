import os
import pandas as pd
from datasets import load_dataset, Dataset

RAW_DATASET_PATH = "src/data/raw/imdb_dataset.csv"
PROCESSED_PATH = "src/data/processed/imdb_hf"

def preprocess_dataset():
    """Preprocess IMDB dataset, balance classes, and save in Hugging Face format."""
    
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"Dataset not found at {RAW_DATASET_PATH}. Cannot preprocess.")
        return

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    df_imdb = pd.read_csv(RAW_DATASET_PATH)
    df_imdb['sentiment'] = df_imdb['sentiment'].map({'positive': 2, 'negative': 0})

    dataset = load_dataset("tweet_eval", "sentiment")
    neutral_samples = [x["text"] for x in dataset["train"] if x["label"] == 1]
    df_neutral = pd.DataFrame({"review": neutral_samples, "sentiment": [1] * len(neutral_samples)})

    df_final = pd.concat([df_imdb, df_neutral]).reset_index(drop=True)
    min_samples = df_final['sentiment'].value_counts().min()
    df_balanced = df_final.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(n=min_samples, random_state=42))

    dataset = Dataset.from_pandas(df_balanced[['review', 'sentiment']])
    split_dataset = dataset.train_test_split(test_size=0.2)

    split_dataset["train"].save_to_disk(f"{PROCESSED_PATH}/train")
    split_dataset["test"].save_to_disk(f"{PROCESSED_PATH}/test")

    print("Dataset preprocessed and saved.")

if __name__ == "__main__":
    preprocess_dataset()
