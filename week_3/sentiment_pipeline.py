from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import os

def load_imdb_data():
    dataset = load_dataset("imdb")
    return dataset

def preprocess_data(dataset, tokenizer):
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding='max_length', max_length=512)
    
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])    
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

def train_model(tokenized_datasets, tokenizer):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model, trainer

def save_model(model, tokenizer, path="sentiment_model"):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=-1).item()
    return {0: "negative", 1: "positive"}[prediction]

if __name__ == "__main__":
    dataset = load_imdb_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_datasets = preprocess_data(dataset, tokenizer)
    model, trainer = train_model(tokenized_datasets, tokenizer)
    
    results = trainer.evaluate()
    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")

    save_model(model, tokenizer)

    example = "The plot was weak and characters were uninteresting."
    prediction = predict_sentiment(example, model, tokenizer)
    print(f"\nInput: {example}")
    print(f"Predicted sentiment: {prediction}")
