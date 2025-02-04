# Sentiment Analysis AI


This is a **Sentiment Analysis Model** using both:
- **Baseline Model** (TF-IDF + Logistic Regression)
- **Fine-Tuned Transformer** (DistilBERT)

It classifies text as **POSITIVE** or **NEGATIVE**.

## Features
- **Two Models**: Classical ML (`baseline`) & Deep Learning (`transformer`)
- Supports **GPU acceleration (CUDA)**
- Runs on a **FastAPI API** (coming soon)
- **High Test Coverage (85%+) with `pytest`**
- **Dockerized for Production**  
- **CI/CD Planned with Kubernetes & Vercel**

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
- Model will be saved in `models/baseline_model/baseline_model.pkl`
- TF-IDF vectorizer saved in `models/baseline_model/tfidf_vectorizer.pkl`

### Train the Transformer Model (DistilBERT)
```
python src/training/train_transformer.py
```
- Model will be saved in `models/transformer_model/transformer_model.pkl`
- Tokenizer saved in `models/transformer_model/transformer_tokenizer.pkl`

### Train the Fine-Tuned Transformer Model (DistilBERT)
```
python src/training/train_transformer.py
```
- Model will be saved in `models/transformer_finetuned`

### Evaluate Models
```
python src/evaluation/evaluate_baseline.py
python src/evaluation/evaluate_transformer.py
python src/evaluation/compare_models.py
```

## Running Tests
To verify everything works:
```
pytest tests/
```

Check coverage:
```
pytest --cov-report term-missing --cov=src tests/
```

To check the HTML file:
```
pytest --cov=src --cov-report=html tests
```

## API Deployment (Coming Soon)
- I'll **containerize the API with Docker**
- Deploy to Vercel and Kubernetes
- Implement CI/CD pipeline with automated tests

## Contributing
Feel free to open a PR!

## License
MIT License - **Feel free to use and modify!**