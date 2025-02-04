from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

dataset = load_from_disk("data/processed/imdb_hf")

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(8000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(2000))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)

dataset = dataset.rename_column("sentiment", "label")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="models/transformer_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("models/transformer_finetuned")
tokenizer.save_pretrained("models/transformer_finetuned")

print("Fine-Tuning Completed & Model Saved!")