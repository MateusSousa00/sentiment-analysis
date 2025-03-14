# Sentiment Analysis AI

A **high-performance Sentiment Analysis Model** that classifies text as **POSITIVE** or **NEGATIVE**, powered by:
- **Baseline Model** (TF-IDF + Logistic Regression)
- **Fine-Tuned Transformer** (DistilBERT)

## Features

- **Dual Model Support**: Choose between classical ML (`baseline`) & deep learning (`transformer`)
- **Fine-Tuned Transformer Model** (DistilBERT) hosted on Hugging Face
- **FastAPI API** (Dockerized & CI/CD Automated)
- **GPU Acceleration** (CUDA Support) for Transformer Inference
- **High Test Coverage** (91%+) with `pytest`
- **CI/CD with Docker & Vercel Deployment**

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
python src/training/train_baseline_model.py
```
- Outputs saved to `src/models/baseline_model/`

### Train the Fine-Tuned Transformer Model (DistilBERT)
```
python src/training/train_transformer.py
```
- Outputs saved to `src/models/transformer_finetuned/`

### Evaluate Models
```
python src/evaluation/evaluate_baseline.py
python src/evaluation/evaluate_transformer.py
python src/evaluation/compare_models.py
```

## Running Tests
Run unit tests:
```
pytest tests/
```

Check test coverage:
```
pytest --cov=src --cov-report=term-missing tests/
```

Generate HTML coverage report:
```
pytest --cov=src --cov-report=html tests
```

## API Deployment

### Run Locally with Docker

```
docker build -t sentiment-analysis-api .
docker run -p 8000:8000 sentiment-analysis-api
```

### Deploying with CI/CD

- Docker Image Build & Push
- Auto-Deploy on Vercel
- CI/CD Automated with GitHub Actions


## Contributing
Feel free to open a PR!

## License
MIT License - **Feel free to use and modify!**