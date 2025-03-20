import os
import joblib
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

env_file = ".env.production" if os.getenv("ENVIRONMENT") == "production" else ".env"
load_dotenv(dotenv_path=env_file)

MODEL_PATH = os.getenv("MODEL_PATH")
BASELINE_MODEL_PATH = os.getenv("BASELINE_MODEL_PATH")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH")

def find_model_checkpoint(model_path):
    """ Procura pelo primeiro checkpoint válido dentro do diretório do modelo. """
    for root, dirs, files in os.walk(model_path):
        if "pytorch_model.bin" in files or "model.safetensors" in files:
            return root
    return None

MODEL_CHECKPOINT_PATH = find_model_checkpoint(MODEL_PATH)

if MODEL_CHECKPOINT_PATH is None:
    raise FileNotFoundError(f"No model found on{MODEL_CHECKPOINT_PATH}!")

baseline_model = joblib.load(BASELINE_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def load_transformer_model():
    """Carrega o modelo transformer corretamente, independentemente do caminho do checkpoint."""
    print(f"📌 Usando o checkpoint em: {MODEL_CHECKPOINT_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)

    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def predict_sentiment(text: str, model_type: str = "baseline"):
    """Predict sentiment using either the baseline or transformer model."""

    if not isinstance(text, str) or not text.strip():
        return {"sentiment": "Neutral", "confidence": 0.50}

    strong_negatives = ["hate", "horrible", "disgust", "despise", "worst", "terrible", "awful", "trash"]
    
    if any(word in text.lower() for word in strong_negatives):
        return {"sentiment": "Negative", "confidence": 0.95}

    if model_type == "transformer":
        label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Positive", "LABEL_2": "Neutral"}
        
        transformer_model = load_transformer_model()
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

