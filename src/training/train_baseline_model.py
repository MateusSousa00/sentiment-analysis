import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from datasets import load_from_disk

# Load the correct train split
train_dataset = load_from_disk("src/data/processed/imdb_hf/train")
df = pd.DataFrame(train_dataset)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words=None,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Balance Dataset (Handling Class Imbalance)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf, y_train)

# Train Model for 3-Class Classification
model = LogisticRegression(solver="lbfgs", max_iter=500)
model.fit(X_train_resampled, y_train_resampled)

# Save Model & Vectorizer
joblib.dump(model, "src/models/baseline_model/baseline_model.pkl")
joblib.dump(vectorizer, "src/models/baseline_model/tfidf_vectorizer.pkl")

print("Baseline model trained and saved successfully.")
