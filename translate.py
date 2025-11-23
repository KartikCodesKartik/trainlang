import numpy as np
import torch
from torch import nn
import argparse
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class BeamSearchTranslator:
    def __init__(self, model_name, beam_width=5, max_length=50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.beam_width = beam_width
        self.max_length = max_length

    def translate(self, input_ids):
        beam_output = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_beams=self.beam_width,
            length_penalty=0.6,
            early_stopping=True
        )
        return beam_output

def load_model(model_name):
    return BeamSearchTranslator(model_name)

def translate_text(translator, text):
    input_ids = translator.tokenizer(text, return_tensors='pt', padding=True).input_ids
    beam_output = translator.translate(input_ids)
    return translator.tokenizer.decode(beam_output[0], skip_special_tokens=True)

def translate_file(translator, file_path):
    translations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            translated_text = translate_text(translator, line.strip())
            translations.append(translated_text)
    return translations

def interactive_mode(translator):
    print("Enter 'exit' to quit.")
    while True:
        text = input("Input text: ")
        if text.lower() == 'exit':
            break
        translated_text = translate_text(translator, text)
        print(f'Translated: {translated_text}')

def main():
    parser = argparse.ArgumentParser(description='Beam Search Translation Inference')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--input', type=str, help='Input file for batch translation')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive mode')

    args = parser.parse_args()
    translator = load_model(args.model)

    if args.interactive:
        interactive_mode(translator)
    elif args.input:
        start_time = time.time()
        translations = translate_file(translator, args.input)
        end_time = time.time()
        for translation in translations:
            print(translation)
        print(f'Translation time: {end_time - start_time:.2f} seconds')
    else:
        print('Please provide an input file or enable interactive mode.')

if __name__ == '__main__':
    main()