import torch
import numpy as np
import joblib
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

dataset = load_from_disk("data/processed/imdb_hf")
test_data = dataset["test"].shuffle(seed=42).select(range(2000))

model_path = "models/transformer_finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def batch_predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return prediction

y_true = []
y_pred = []


for i in tqdm(range(0, len(test_data), 32), desc="Evaluating", ncols=80):
    batch = [test_data[j] for j in range(i, min(i + 32, len(test_data)))]
    texts = [example["review"] for example in batch]
    labels = [example["sentiment"] for example in batch]
    
    preds = batch_predict(texts)
    
    y_true.extend(labels)
    y_pred.extend(preds)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

print("\n Model Performance:")
print(f" Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1-score:  {f1:.4f}")

results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

joblib.dump(results, "models/transformer_finetuned/evaluation_results.pkl")
print("\n Evaluation results saved!")