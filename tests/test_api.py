import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_baseline():
    response = client.get("/predict/", params={"text": "I love AI!", "model_type": "baseline"})
    assert response.status_code == 200
    assert "sentiment" in response.json()
    
def test_api_transformer():
    response = client.get("/predict/", params={"text": "I hate bugs.", "model_type": "transformer"})
    assert response.status_code == 200
    assert "sentiment" in response.json()

def test_api_invalid_model_type():
    response = client.get("/predict/", params={"text": "Hello", "model_type": "invalid_model"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid model type. Choose 'baseline' or 'transformer'."

def test_api_empty_text():
    response = client.get("/predict/", params={"text": "", "model_type": "baseline"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Input text cannot be empty."

def test_api_invalid_request():
    response = client.get("/predict/", params={"wrong_key": "Hello"})
    assert response.status_code == 422

if __name__ == "__main__":
    pytest.main()