# Sentiment Analysis AI

This is a **Sentiment Analysis model** using Hugging Face's `transformers` and `Pytorch`.
It classifies text as **POSITIVE** or **NEGATIVE**.


## Features
- Pretrained model: `distilbert-base-uncased-finetuned-sst-2-english`
- Supports **GPU acceleration (CUDA)**
- Runs on a **FastAPI API** (soon)
- Includes **unit tests with `pytest`**


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

4. **Run the model**
```
python src/train_model.py
```


## Running Tests
To verify everything works:
```
pytest tests/
```


## Example Usage
```
from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")
print(sentiment_model("I love AI!"))
# Output: [{'label': 'POSITIVE', 'score': 0.9999}]
```


## Upcoming Features
- Fine-tuned model on custom data
- Deploy as an API using FastAPI
- Web interface for sentiment analysis

## Contributing
Feel free to open a PR!


## License
MIT License - **Feel free to use and modify!**