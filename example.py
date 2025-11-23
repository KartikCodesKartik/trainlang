# example.py

# Import the necessary libraries
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load the model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-ru-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def simple_translation(text):
    """Translate a single text input."""
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def batch_translation(texts):
    """Translate a batch of text inputs."""
    translated = model.generate(**tokenizer(texts, return_tensors="pt", padding=True, truncation=True))
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

def interactive_translation():
    """Interactive mode for continuous translation."""
    while True:
        text = input("Enter text to translate (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        print("Translation:", simple_translation(text))

def model_details():
    """Print model architecture details."""
    print("Model Architecture:\n", model)

def evaluate_example():
    """Example evaluation of the translation model."""
    examples = [
        "Привет, как дела?",  # "Hello, how are you?"
        "Где находится библиотека?"  # "Where is the library?"
    ]
    translations = batch_translation(examples)
    for original, translated in zip(examples, translations):
        print(f"Original: {original} -> Translated: {translated}")

if __name__ == "__main__":
    print("Simple Translation Example:", simple_translation("Как дела?"))  # "How are you?"
    print("Batch Translation Example:", batch_translation(["Доброго утра!", "Как твои дела?"]))  # ["Good morning!", "How are you?"]
    print("Model Details:")
    model_details()
    print("Running Interactive Translation Mode:")
    interactive_translation()
    print("Evaluation Examples:")
    evaluate_example()
