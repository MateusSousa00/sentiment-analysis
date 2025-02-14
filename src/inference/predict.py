import joblib
from transformers import pipeline
from src.inference.text_utils import is_textual_input, is_question
from src.inference.sentiment_analysis import classify_baseline, classify_transformer

def predict_sentiment(text, model_type="baseline"):
    """Handles sentiment prediction by routing to the correct model."""
    
    if not is_textual_input(text):
        return {"sentiment": "Neutral", "confidence": 0.50}

    if is_question(text):
        return {"sentiment": "Neutral", "confidence": 0.50}

    if len(text.strip()) <= 2:
        return {"sentiment": "Neutral", "confidence": 0.50}

    if model_type == "baseline":
        return classify_baseline(text)

    if model_type == "transformer":
        return classify_transformer(text)

    raise ValueError("Invalid model type. Choose 'baseline' or 'transformer'.")
