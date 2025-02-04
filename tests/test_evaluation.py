import joblib
import pytest
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_from_disk

# Load
baseline_model = joblib.load("models/baseline_model/baseline_model.pkl")
vectorizer = joblib.load("models/baseline_model/tfidf_vectorizer.pkl")

dataset = load_from_disk("data/processed/imdb_hf")
test_data = dataset["test"]

random.seed(42)
test_sample = random.sample(list(test_data), 100)

test_texts = [entry["review"] for entry in test_sample]
test_labels = [entry["sentiment"] for entry in test_sample]

# Test 1: Baseline Model Accuracy
def test_baseline_model_accuracy():
    transformed_texts = vectorizer.transform(test_texts)
    predictions = baseline_model.predict(transformed_texts)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\n Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.80

# Test 2: Precision Score
def test_baseline_model_precision():
    transformed_texts = vectorizer.transform(test_texts)
    predictions = baseline_model.predict(transformed_texts)
    precision = precision_score(test_labels, predictions)

    print(f"\n Precision: {precision:.4f}")
    
    assert precision > 0.75

# Test 3: Recall Score
def test_baseline_model_recall():
    transformed_texts = vectorizer.transform(test_texts)
    predictions = baseline_model.predict(transformed_texts)
    recall = recall_score(test_labels, predictions)

    print(f"\n Recall: {recall:.4f}")
    
    assert recall > 0.75

# Test 4: F1 Score
def test_baseline_model_f1_score():
    transformed_texts = vectorizer.transform(test_texts)
    predictions = baseline_model.predict(transformed_texts)
    f1 = f1_score(test_labels, predictions)

    print(f"\n F1 Score: {f1:.4f}")
    
    assert f1 > 0.75
    
if __name__ == "__main__":
    pytest.main()
