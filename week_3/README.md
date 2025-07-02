# Sentiment Analysis on IMDb using BERT

This assignment implements a sentiment analysis pipeline using a pre-trained BERT model fine-tuned on the IMDb movie review dataset. It uses Hugging Face's `transformers` and `datasets` libraries to perform binary classification (positive/negative).

---

## Files

- `sentiment_pipeline.py`: Python script that loads data, tokenizes it, fine-tunes BERT and performs sentiment prediction.
- `report.pdf`: Final report pdf with explanation and list of challenges faced.
- `README.md`: This file.

---

## How to Run

### 1. Install Dependencies

```bash
pip install transformers datasets scikit-learn torch
```

### 2. Run the Script

```bash
python sentiment_pipeline.py
```

> 💡 Note: The full IMDb dataset (~25k reviews) is automatically downloaded. A GPU is recommended for faster training (e.g., Google Colab, Kaggle or CUDA-supported machine).

---

## ⚙️ Model Configuration

- Model: `bert-base-uncased`
- Max Token Length: 512
- Training Batch Size: 8
- Epochs: 2
- Optimizer: AdamW (used via Hugging Face Trainer)

---

## ✅ Features

- Load IMDb dataset using Hugging Face `datasets`
- Tokenize input text using BERT tokenizer with truncation and padding
- Fine-tune BERT model on binary sentiment labels
- Evaluate using Accuracy and F1-Score
- Save model and tokenizer for future inference
- Run live sentiment prediction on custom text

---

## 🧠 Inference Example

```python
example = "The movie was absolutely fantastic!"
print(predict_sentiment(example, model, tokenizer))
# Output: positive
```

---

## 📘 Report

A detailed explanation of the pipeline, along with difficulties faced and their resolutions, is included in `report.pdf`.
