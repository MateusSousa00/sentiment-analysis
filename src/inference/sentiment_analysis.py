import joblib
from transformers import pipeline
from src.inference.text_utils import is_neutral_with_textblob, is_neutral_with_vader

# Load models
baseline_model = joblib.load("src/models/baseline_model/baseline_model.pkl")
vectorizer = joblib.load("src/models/baseline_model/tfidf_vectorizer.pkl")

transformer_model = pipeline("sentiment-analysis", model="src/models/transformer_finetuned")

def classify_baseline(text):
    """Predict sentiment using the baseline model."""
    text_tfidf = vectorizer.transform([text])
    prediction = baseline_model.predict(text_tfidf)[0]
    confidence = max(baseline_model.predict_proba(text_tfidf)[0])

    if is_neutral_with_textblob(text) or is_neutral_with_vader(text):
        return {"sentiment": "Neutral", "confidence": confidence}

    return {"sentiment": "Positive" if prediction == 1 else "Negative", "confidence": confidence}

def classify_transformer(text):
    """Predict sentiment using the transformer model."""
    result = transformer_model(text)[0]
    confidence = round(result["score"], 4)

    if is_neutral_with_textblob(text) or is_neutral_with_vader(text):
        return {"sentiment": "Neutral", "confidence": confidence}

    return {"sentiment": "Positive" if result["label"] == "LABEL_1" else "Negative", "confidence": confidence}
