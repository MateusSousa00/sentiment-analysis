import sys
import os
import pytest
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    """Ensure the baseline model and vectorizer exist before running tests."""
    model_path = "src/models/baseline_model/baseline_model.pkl"
    vectorizer_path = "src/models/baseline_model/tfidf_vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("⚡️ Training baseline model before running tests...")
        subprocess.run(["python", "src/training/train_baseline_model.py"], check=True)