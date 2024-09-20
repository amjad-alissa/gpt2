import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def evaluate_model(model_path, original_text):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(original_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    model_path = './gpt2_simplification_model'
    original_text = "Das ist ein komplexer Satz."
    simplified_text = evaluate_model(model_path, original_text)
    print(f"Simplified Text: {simplified_text}")
