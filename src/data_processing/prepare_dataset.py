import pandas as pd
from datasets import Dataset

df = pd.read_csv("data/raw/imdb_dataset.csv")

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

dataset = Dataset.from_pandas(df[['review', 'sentiment']])

dataset = dataset.train_test_split(test_size=0.2)

dataset.save_to_disk("data/processed/imdb_hf")

print("IMDB dataset successfully prepared for training!")
