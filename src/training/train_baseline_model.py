import os
import pandas as pd
import joblib
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

PROCESSED_PATH = "src/data/processed/imdb_hf"
MODEL_PATH = "src/models/baseline_model"

def train_baseline_model():
    """Train baseline model using Logistic Regression and save."""
    
    if not os.path.exists(f"{PROCESSED_PATH}/train"):
        print(f"Processed dataset not found at {PROCESSED_PATH}/train. Cannot train model.")
        return

    os.makedirs(MODEL_PATH, exist_ok=True)

    train_dataset = load_from_disk(f"{PROCESSED_PATH}/train")
    df = pd.DataFrame(train_dataset)

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf, y_train)

    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(X_train_resampled, y_train_resampled)

    joblib.dump(model, f"{MODEL_PATH}/baseline_model.pkl")
    joblib.dump(vectorizer, f"{MODEL_PATH}/tfidf_vectorizer.pkl")

    print("Baseline model trained and saved.")

if __name__ == "__main__":
    train_baseline_model()
