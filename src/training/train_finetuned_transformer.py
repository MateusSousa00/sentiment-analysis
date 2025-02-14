from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# Load dataset as DatasetDict
dataset = DatasetDict.load_from_disk("src/data/processed/imdb_hf")

# Access train and test splits
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Reduce dataset size for better performance (optional)
train_dataset = train_dataset.shuffle(seed=42).select(range(20000))  # Use 20k samples for training
test_dataset = test_dataset.shuffle(seed=42).select(range(4000))  # Use 4k samples for testing

# Load model & tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Send model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Rename target column
train_dataset = train_dataset.rename_column("sentiment", "label")
test_dataset = test_dataset.rename_column("sentiment", "label")

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments optimized for GPU
training_args = TrainingArguments(
    output_dir="src/models/transformer_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision for faster training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

print("Training script is ready. Run this command tonight:")
print("python src/training/train_finetuned_transformer.py")
