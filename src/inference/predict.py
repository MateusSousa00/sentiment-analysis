import joblib
from transformers import pipeline

def predict_sentiment(text, model_type="baseline"):
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("Input text cannot be empty.")
    
    if model_type == "baseline":
        model = joblib.load("models/baseline_model/baseline_model.pkl")
        vectorizer = joblib.load("models/baseline_model/tfidf_vectorizer.pkl")
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]

        return {"sentiment": "Positive" if prediction == 1 else "Negative"}

    elif model_type == "transformer":
        model = joblib.load("models/transformer_model/transformer_model.pkl")
        tokenizer = joblib.load("models/transformer_model/transformer_tokenizer.pkl")
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        result = sentiment_pipeline(text)[0]

        return {"sentiment": result["label"], "confidence": result["score"]}

    else:
        raise ValueError("Invalid model type. Choose 'baseline' or 'transformer'.")
    
if __name__ == "__main__":
    print(predict_sentiment("I loved the movie! Best film ever.", model_type="baseline"))
    print(predict_sentiment("Worst experience. I will never watch this again.", model_type="transformer"))