from fastapi import FastAPI, HTTPException
from src.predict import predict_sentiment

app = FastAPI()

@app.get("/predict/")
def get_sentiment(text: str, model_type: str = "baseline"):
    try:
        sentiment = predict_sentiment(text, model_type)
        return {"text": text, "sentiment": sentiment, "model_used": model_type}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)