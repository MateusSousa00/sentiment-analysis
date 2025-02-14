from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.api.models import predict_sentiment

router = APIRouter()

class SentimentRequest(BaseModel):
    text: str
    model_type: str = "baseline"
    
@router.post("/predict/")
def predict(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        result = predict_sentiment(request.text, request.model_type)
        return {"sentiment": result["sentiment"], "confidence": result["confidence"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))