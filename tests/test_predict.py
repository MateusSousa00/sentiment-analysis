import pytest
from src.inference.predict import predict_sentiment

# Test Baseline Model (Logistic Regression)
def test_predict_baseline_positive():
    result = predict_sentiment("I love this movie!", model_type="baseline")
    assert isinstance(result, dict)
    assert result["sentiment"] == "Positive"

def test_predict_baseline_negative():
    result = predict_sentiment("I hate this movie!", model_type="baseline")
    assert isinstance(result, dict)

    print(f"🔍 Baseline Prediction for 'I hate this movie!': {result}")
    assert result["sentiment"] == "Negative"

# Test Transformer Model (DistilBERT)
def test_predict_transformer_positive():
    result = predict_sentiment("I love this movie!", model_type="transformer")
    assert isinstance(result, dict)
    assert result["sentiment"] == "POSITIVE"
    assert result["confidence"] >= 0.8

def test_predict_transformer_negative():
    result = predict_sentiment("I hate this movie!", model_type="transformer")
    assert isinstance(result, dict)
    assert result["sentiment"] == "NEGATIVE"
    assert result["confidence"] >= 0.8

# ✅ Test Empty Input Handling
def test_predict_empty():
    with pytest.raises(ValueError):
        predict_sentiment("", model_type="baseline")

if __name__ == "__main__":
    pytest.main()
