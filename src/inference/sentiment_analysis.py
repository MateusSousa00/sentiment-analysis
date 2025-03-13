import joblib
from transformers import pipeline
from src.inference.text_utils import is_neutral_with_textblob, is_neutral_with_vader

# Load models
baseline_model = joblib.load("src/models/baseline_model/baseline_model.pkl")
vectorizer = joblib.load("src/models/baseline_model/tfidf_vectorizer.pkl")

transformer_model = pipeline("sentiment-analysis", model="src/models/transformer_finetuned")

def classify_baseline(text):
    """Predict sentiment using the baseline model."""
    transformed_text = vectorizer.transform([text])
    probs = baseline_model.predict_proba(transformed_text)[0]
    prediction = probs.argmax()  
    confidence = probs[prediction] 
    
    LABEL_MAPPING = {0: "Negative", 1: "Positive", 2: "Neutral"}

    if confidence < 0.45:
        return {"sentiment": "Neutral", "confidence": confidence}

    sentiment = LABEL_MAPPING.get(prediction, "Unknown")
    return {"sentiment": sentiment, "confidence": confidence}

def classify_transformer(text):
    """Predict sentiment using the transformer model."""
    
    result = transformer_model(text)[0]
    confidence = round(result["score"], 4)

    label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    sentiment = label_mapping.get(result["label"], "Unknown")

    if confidence < 0.5:
        return {"sentiment": "Neutral", "confidence": confidence}
    
    if "hate" in text.lower() and sentiment in ["Positive", "Neutral"] and confidence > 0.90:
        sentiment = "Negative"

    return {"sentiment": sentiment, "confidence": confidence}

