import pytest
from unittest import mock
from src.api.models import predict_sentiment

@pytest.fixture
def mock_pipeline():
    """Mock the Hugging Face pipeline for transformer model testing."""
    with mock.patch("src.api.models.pipeline") as mock_pipeline:
        mock_model = mock_pipeline.return_value
        mock_model.return_value = [{"label": "LABEL_1", "score": 0.85}]  # Simulate a positive sentiment
        yield mock_pipeline

def test_predict_sentiment_baseline():
    """Test sentiment prediction using the baseline model."""
    result = predict_sentiment("I love this product!", model_type="baseline")
    assert result["sentiment"] in ["Positive", "Negative", "Neutral"]
    assert 0.5 <= result["confidence"] <= 1.0

def test_predict_sentiment_transformer(mock_pipeline):
    """Test sentiment prediction using the transformer model."""
    result = predict_sentiment("I love this product!", model_type="transformer")
    assert result["sentiment"] == "Positive"
    assert 0.5 <= result["confidence"] <= 1.0

def test_predict_sentiment_fallback_to_huggingface(mock_pipeline):
    """Test model download from Hugging Face when local model is missing."""
    mock_pipeline.side_effect = [Exception("Local model not found"), mock_pipeline.return_value]
    
    result = predict_sentiment("This is fine.", model_type="transformer")
    assert result["sentiment"] == "Positive"  # Since we mocked a positive sentiment
    assert result["confidence"] > 0.8  # Mocked confidence

def test_predict_sentiment_invalid_model_type():
    """Test invalid model type handling."""
    with pytest.raises(ValueError):
        predict_sentiment("I love AI!", model_type="unknown")

def test_predict_sentiment_with_empty_input():
    """Test empty input returns neutral sentiment."""
    result = predict_sentiment("", model_type="baseline")
    assert result["sentiment"] == "Neutral"
    assert result["confidence"] == 0.50

def test_predict_sentiment_with_strong_negative():
    """Test predefined strong negative words get classified correctly."""
    result = predict_sentiment("I hate this!", model_type="baseline")
    assert result["sentiment"] == "Negative"
    assert result["confidence"] == 0.95
