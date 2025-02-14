import joblib
from transformers import pipeline
from src.utils.utils import is_textual_input, is_question, is_neutral_statement

baseline_model = joblib.load("src/models/baseline_model/baseline_model.pkl")
vectorizer = joblib.load("src/models/baseline_model/tfidf_vectorizer.pkl")

transformer_model = pipeline("sentiment-analysis", model="src/models/transformer_finetuned")

def predict_sentiment(text: str, model_type: str = "baseline"):
    if not is_textual_input(text):
        return {"sentiment": "Neutral", "confidence": None}

    if is_question(text) or is_neutral_statement(text):
        return {"sentiment": "Neutral", "confidence": None}

    if model_type == "baseline":
        text_tfidf = vectorizer.transform([text])
        prediction = int(baseline_model.predict(text_tfidf)[0])  # Ensure it's an integer
        probs = baseline_model.predict_proba(text_tfidf)[0]  # Get class probabilities
        confidence = round(max(probs), 4)
        
        strong_negatives = ["hate", "terrible", "disgusting", "awful", "worst", "horrible", "bad", "nasty"]
        if any(word in text.lower() for word in strong_negatives):
            return {"sentiment": "Negative", "confidence": max(confidence, 0.85)} 

        if confidence < 0.55:
            return {"sentiment": "Neutral", "confidence": None}

        sentiment = "Positive" if prediction == 1 else "Negative"
        return {"sentiment": sentiment, "confidence": confidence}

    elif model_type == "transformer":
        result = transformer_model(text)[0]
        label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Positive"}
        sentiment = label_mapping.get(result["label"], "Unknown")
        confidence = round(result["score"], 4)

        if confidence < 0.7:  # Adjusting threshold
            return {"sentiment": "Neutral", "confidence": None}

        return {"sentiment": sentiment, "confidence": confidence}

    else:
        raise ValueError("Invalid model type. Choose 'baseline' or 'transformer'.")
