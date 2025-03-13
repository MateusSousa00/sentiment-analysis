from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# Load dataset properly (train & test are separate)
train_dataset = Dataset.load_from_disk("src/data/processed/imdb_hf/train")
test_dataset = Dataset.load_from_disk("src/data/processed/imdb_hf/test")

# Reduce dataset size for better performance (optional)
train_dataset = train_dataset.shuffle(seed=42).select(range(min(20000, len(train_dataset)))) # Use up to 20k samples
test_dataset = test_dataset.shuffle(seed=42).select(range(min(4000, len(test_dataset)))) # Use up to 4k samples

# Load model & tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Send model to GPU if available
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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(), 
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Start Training
print("\n🚀 Training Started! This may take several hours...\n")
trainer.train()

# Save Model & Tokenizer
model.save_pretrained("src/models/transformer_finetuned")
tokenizer.save_pretrained("src/models/transformer_finetuned")

print("\n Training Complete! Model saved at 'src/models/transformer_finetuned'.")
