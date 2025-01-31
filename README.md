# Sentiment Analysis AI

This is a **Sentiment Analysis model** using Hugging Face's `transformers` and `Pytorch`.
It classifies text as **POSITIVE** or **NEGATIVE**.


## Features
- Pretrained model: `distilbert-base-uncased-finetuned-sst-2-english`
- Supports **GPU acceleration (CUDA)**
- Runs on a **FastAPI API**
- Includes **unit tests with `pytest`**
- Saves and loads trained models for inference

## Instalation

1. **Clone the repo**
```
git clone https://github.com/mateussousa00/sentiment-analysis.git
cd sentiment-analysis
```
2. **Set up a virtual environment**
```
python -m venv venv
source venv/bin/activate # For Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```
pip install -r requirements.txt
```

## Training Models

### Train the Baseline Model (Logistic Regression + TF-IDF)
```
python src/train_baseline_model.py
```
- Model will be saved in `models/sentiment_model.pkl`
- TF-IDF vectorizer saved in `models/tfidf_vectorizer.pkl`

### Train the Transformer Model (DistilBERT)
```
python src/train_transformer_model.py
```
- Model will be saved in `models/transformer_model.pkl`
- Tokenizer saved in `models/transformer_tokenizer.pkl`

## Running Predictions

### Predict using the **Baseline Model**
```
python src/predict.py
```
**Example:**
```
from src.predict import predict_sentiment
print(predict_sentiment("I love AI!", model_type="baseline"))
# Expected output: Positive
```

### Predict using the **Transformer Model**
```
python src/predict.py
```
**Example:**
```
from src.predict import predict_sentiment
print(predict_sentiment("I love AI!", model_type="transformer"))
#Expected output: Sentiment: POSITIVE (Confidence: 99.94%)
```

## Running FastAPI Server

1. Start the API server
```
uvicorn api.main:app --reload
```

2. Open your browser and test:
```
http://127.0.0.1:8000/predict/?text=I%20love%20this!&model_type=transformer
```

Or test via **cURL**:
```
curl "http://127.0.0.1:8000/predict/?text=I%20love%20this!&model_type=transformer"
```

## Running Tests
To verify everything works:
```
pytest tests/
```

## Upcoming Features
- Fine-tuned model on custom data
- Web interface for sentiment analysis

## Contributing
Feel free to open a PR!


## License
MIT License - **Feel free to use and modify!**