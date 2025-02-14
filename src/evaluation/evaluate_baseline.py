import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk

baseline_model = joblib.load("src/models/baseline_model/baseline_model.pkl")
vectorizer = joblib.load("src/models/baseline_model/tfidf_vectorizer.pkl")

dataset = load_from_disk("src/data/processed/imdb_hf")
test_data = dataset["test"].shuffle(seed=42).select(range(2000))

X_test_tfidf = vectorizer.transform([example["review"] for example in test_data])
y_true = [example["sentiment"] for example in test_data]

y_pred = baseline_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

print("\n Baseline Model Performance:")
print(f" Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1-score:  {f1:.4f}")

results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}
joblib.dump(results, "src/models/baseline_model/evaluation_results.pkl")

print("\n Baseline model evaluation saved!")