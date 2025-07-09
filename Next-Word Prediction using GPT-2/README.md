# Next Word Prediction with GPT-2

This project fine-tunes a GPT-2 model on the WikiText-2 dataset for next-word prediction and provides an interactive Gradio-based interface to test predictions.

```
next_word_prediction/
├── data/                     # Tokenized WikiText-2 dataset
│   └── wikitext/
├── model/                    # Fine-tuned GPT-2 checkpoints
│   └── checkpoints/
├── notebooks/
│   └── train_gpt2.ipynb      # Notebook for preprocessing & training
├── app/
│   └── demo.py               # Gradio interface for inference
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/next_word_prediction.git
   cd next_word_prediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Training

- Run the notebook:

   ```bash
   jupyter notebook notebooks/train_gpt2.ipynb
   ```

- This notebook will:
   - Download and tokenize WikiText-2
   - Fine-tune GPT-2
   - Save the model and tokenizer to `model/checkpoints/`

---

## Run the Gradio App

```bash
python app/demo.py
```

- Access the interface in your browser at `http://127.0.0.1:7860`

---

## Example

**Input Prompt:**
```
Hey, How are
```

**Predicted Output:**
```
Hey, How are you
```

---

## Dependencies

See `requirements.txt`:

```
transformers==4.53.1
datasets
evaluate
torch
numpy
gradio
```

---

## Future Ideas

- Support for top-k sampling control
- Compare performance with LSTM baseline
- Streamlit alternative frontend
- Deploy to Hugging Face Spaces

---
