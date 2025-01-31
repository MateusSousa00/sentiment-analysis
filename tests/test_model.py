import pytest
from src.predict import predict_sentiment

def test_baseline_prediction():
    result = predict_sentiment("I love AI!", model_type="baseline")
    assert result in ["Positive", "Negative"]
    
def test_transformer_prediction():
    result = predict_sentiment("I hate bugs.", model_type="transformer")
    assert isinstance(result, str)
    
def test_empty_input():
    with pytest.raises(ValueError):
        predict_sentiment("", model_type="baseline")
        
def test_invalid_input():
    with pytest.raises(ValueError):
        predict_sentiment(12345, model_type="transformer")
        
def test_unexpected_output():
    result = predict_sentiment("I love AI!", model_type="baseline")
    assert result in ["Positive", "Negative"], f"Unexpected output: {result}"
    
if __name__ == "__main__":
    pytest.main()