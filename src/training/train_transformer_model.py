from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib

def train_transformer_model():
    print("Initializing Transformer model training...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    joblib.dump(model, 'models/transformer_model/transformer_model.pkl')
    joblib.dump(tokenizer, 'models/transformer_model/transformer_tokenizer.pkl')
    print("Transformer model trained and saved successfully!")
    
if __name__ == "__main__":
    try:
        train_transformer_model()
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("CUDA out of memory! Try reducing batch size or using CPU instead.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error ocurred during Transformer training: {e}")