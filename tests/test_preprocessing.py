import pytest
from src.preprocessing.text_cleaning import preprocess_text

def test_preprocess_text():
    text = "I LOVED this movie!!! <br> 10/10 would recommend it!"
    cleaned_text = preprocess_text(text)
    
    assert isinstance(cleaned_text, str)
    assert "loved" in cleaned_text
    assert "<br>" not in cleaned_text
    
def test_preprocess_empty_string():
    text = ""
    cleaned_text = preprocess_text(text)
    assert cleaned_text == ""
    
def test_preprocess_special_chars():
    text = "!@#$%^&*()"
    cleaned_text = preprocess_text(text)
    assert cleaned_text == ""
    
def test_preprocess_large_text():
    text = "Good " * 10000
    cleaned_text = preprocess_text(text)
    assert isinstance(cleaned_text, str)
    
if __name__ == "__main__":
    pytest.main()