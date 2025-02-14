import pytest
from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def test_positive_sentiment():
    """Test if positive text is classified correctly"""
    result = sentiment_model("I love AI!")[0]
    assert result["label"] == "POSITIVE"
    assert result["score"] > 0.95

def test_negative_sentiment():
    """Test if negative text is classified correctly"""
    result = sentiment_model("I hate shoes.")[0]
    assert result["label"] == "NEGATIVE"
    assert result["score"] > 0.95

def test_neutral_sentiment():
    """Test a more neutral statement"""
    result = sentiment_model("This is an average fan.")[0]
    assert result["label"] in ["POSITIVE", "NEGATIVE"]

def test_model_performance():
    """Ensure the model runs within 1 second"""
    import time
    start_time = time.time()
    sentiment_model("AI is cool!")
    end_time = time.time()
    assert end_time - start_time < 1

if __name__ == "__main__":
    pytest.main()