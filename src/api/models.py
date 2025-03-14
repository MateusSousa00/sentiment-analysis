import joblib
from transformers import pipeline
from src.utils.utils import is_textual_input, is_question, is_neutral_statement

baseline_model = joblib.load("src/models/baseline_model/baseline_model.pkl")
vectorizer = joblib.load("src/models/baseline_model/tfidf_vectorizer.pkl")
MODEL_PATH = "src/models/transformer_finetuned"
HUGGINGFACE_MODEL = "Mateussousa00/sentiment-analysis-model"

def predict_sentiment(text: str, model_type: str = "baseline"):
    """Predict sentiment using either the baseline or transformer model."""

    if not isinstance(text, str) or not text.strip():
        return {"sentiment": "Neutral", "confidence": 0.50}

    strong_negatives = ["hate", "horrible", "disgust", "despise", "worst", "terrible", "awful", "trash"]
    
    if any(word in text.lower() for word in strong_negatives):
        return {"sentiment": "Negative", "confidence": 0.95}

    if model_type == "transformer":
        label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Positive", "LABEL_2": "Neutral"}
        
        try:
            transformer_model = pipeline("sentiment-analysis", model=MODEL_PATH)
        except:
            print("⚠️ Model not found locally. Downloading from Hugging Face...")
            transformer_model = pipeline("sentiment-analysis", model=HUGGINGFACE_MODEL)

        result = transformer_model(text)[0]
        sentiment = label_mapping.get(result["label"], "Unknown")
        confidence = round(result["score"], 4)

        if confidence < 0.65:
            return {"sentiment": "Neutral", "confidence": confidence}

        return {"sentiment": sentiment, "confidence": confidence}

    elif model_type == "baseline":
        transformed_text = vectorizer.transform([text])
        prediction = baseline_model.predict(transformed_text)[0]
        probs = baseline_model.predict_proba(transformed_text)[0]
        confidence = round(max(probs), 4)

        sentiment_labels = {0: "Negative", 1: "Positive", 2: "Neutral"}
        return {"sentiment": sentiment_labels[prediction], "confidence": confidence}
    
    else:
        raise ValueError("Invalid model type. Choose 'baseline' or 'transformer'.")

