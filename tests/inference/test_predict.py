import pytest
from src.inference.predict import predict_sentiment

# Test Baseline Model (Logistic Regression)
def test_predict_baseline_positive():
    result = predict_sentiment("I love this movie!", model_type="baseline")
    assert isinstance(result, dict)
    assert result["sentiment"] in ["Positive", "Neutral"], f"Unexpected sentiment value: {result['sentiment']}"

def test_predict_baseline_negative():
    result = predict_sentiment("I hate this movie!", model_type="baseline")
    assert isinstance(result, dict)

    assert result["sentiment"] == "Negative"

def test_predict_transformer_positive():
    result = predict_sentiment("I love this movie!", model_type="transformer")
    assert isinstance(result, dict)
    assert result["sentiment"] in ["Positive", "Neutral"], f"Unexpected sentiment value: {result['sentiment']}"

def test_predict_transformer_negative():
    result = predict_sentiment("I hate this movie!", model_type="transformer")
    assert isinstance(result, dict)
    assert result["sentiment"] == "Negative"
    assert result["confidence"] >= 0.8

def test_predict_empty():
    result = predict_sentiment("", model_type="baseline")
    assert isinstance(result, dict)
    assert result["sentiment"] == "Neutral"
    assert result["confidence"] == 0.50

if __name__ == "__main__":
    pytest.main()
