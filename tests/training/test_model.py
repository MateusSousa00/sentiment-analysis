import pytest
from src.inference.predict import predict_sentiment

def test_baseline_prediction():
    result = predict_sentiment("I love AI!", model_type="baseline")
    assert isinstance(result, dict)
    assert result["sentiment"] in ["Positive", "Neutral"], f"Unexpected sentiment value: {result['sentiment']}"

def test_transformer_prediction():
    result = predict_sentiment("I hate it so much, I can't stand it.", model_type="transformer")
    assert isinstance(result, dict)
    
    assert result["sentiment"] == "Negative", f"Unexpected sentiment value: {result['sentiment']} with confidence {result['confidence']}"

def test_empty_input():
    result = predict_sentiment("", model_type="baseline")
    assert result["sentiment"] == "Neutral"
    assert result["confidence"] == 0.50

def test_invalid_input():
    result = predict_sentiment("   ", model_type="baseline")
    assert result["sentiment"] == "Neutral"
    assert result["confidence"] == 0.50

def test_unexpected_output():
    result = predict_sentiment("I love AI!", model_type="baseline")
    assert isinstance(result, dict)
    assert result["sentiment"] in ["Positive", "Negative", "Neutral"], f"Unexpected sentiment value: {result['sentiment']}"

    
if __name__ == "__main__":
    pytest.main()