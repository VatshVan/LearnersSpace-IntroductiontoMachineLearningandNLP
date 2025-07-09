from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gradio as gr

# Load tokenizer and model from local fine-tuned checkpoint
tokenizer = GPT2Tokenizer.from_pretrained("model/checkpoints/final")
model = GPT2LMHeadModel.from_pretrained("model/checkpoints/final")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# Define prediction function
def predict_next_word(prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1, do_sample=True, top_k=50, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create Gradio interface
demo = gr.Interface(fn=predict_next_word, inputs="text", outputs="text", title="Next Word Predictor")
demo.launch(share=True)
print("\npp is running! Click below to open it in your browser:")
print("Gradio Link will appear above after loading...")
