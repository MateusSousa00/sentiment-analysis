import pandas as pd
from datasets import load_dataset
from datasets import Dataset

# Step 1: Load IMDB Dataset
df_imdb = pd.read_csv("src/data/raw/imdb_dataset.csv")
df_imdb['sentiment'] = df_imdb['sentiment'].map({'positive': 2, 'negative': 0})  # Adjust labels for 3-class classification

# Step 2: Load Neutral Sentences from Hugging Face
dataset = load_dataset("tweet_eval", "sentiment")

# Extract neutral tweets (label = 1)
neutral_samples = [x["text"] for x in dataset["train"] if x["label"] == 1]
print(f"Loaded {len(neutral_samples)} neutral samples from Hugging Face.")

# Convert to DataFrame
df_neutral = pd.DataFrame({"review": neutral_samples, "sentiment": [1] * len(neutral_samples)})

# Step 3: Merge & Balance Dataset
df_final = pd.concat([df_imdb, df_neutral]).reset_index(drop=True)

# Ensure dataset has equal Positive, Neutral, and Negative samples
min_samples = min(df_final['sentiment'].value_counts())
df_balanced = df_final.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(n=min_samples, random_state=42))

print(f"Final dataset size: {len(df_balanced)} (Balanced across 3 labels).")

# Step 4: Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df_balanced[['review', 'sentiment']])

# Step 5: Ensure correct train-test split
split_dataset = dataset.train_test_split(test_size=0.2)

# Step 6: Save both train and test splits separately
split_dataset["train"].save_to_disk("src/data/processed/imdb_hf/train")
split_dataset["test"].save_to_disk("src/data/processed/imdb_hf/test")

print("Dataset updated with proper train-test split and saved!")
