import pytest
from src.inference.predict import predict_sentiment

def test_baseline_prediction():
    result = predict_sentiment("I love AI!", model_type="baseline")
    assert result["sentiment"] == "Positive"
    
def test_transformer_prediction():
    result = predict_sentiment("I hate bugs.", model_type="transformer")
    assert result["sentiment"] == "NEGATIVE"
    
def test_empty_input():
    with pytest.raises(ValueError):
        predict_sentiment("", model_type="baseline")
        
def test_invalid_input():
    with pytest.raises(ValueError):
        predict_sentiment(12345, model_type="transformer")
        
def test_unexpected_output():
    result = predict_sentiment("I love AI!", model_type="baseline")
    assert isinstance(result, dict), f"Unexpected output type: {type(result)}"
    assert result["sentiment"] in ["Positive", "Negative"], f"Unexpected sentiment value: {result['sentiment']}"
    
if __name__ == "__main__":
    pytest.main()