import argparse
import torch
import sacrebleu
import json

# Load the translation model
def load_model(model_path):
    model = torch.load(model_path)
    return model

# Translate the test data
def translate(model, test_data):
    translations = []
    for sentence in test_data:
        translated_sentence = model.translate(sentence)  # Assuming model has a translate method
        translations.append(translated_sentence)
    return translations

# Calculate BLEU scores
def calculate_bleu(reference_data, translated_data):
    bleu = sacrebleu.corpus_bleu(translated_data, [reference_data])
    return bleu.score

# Show example translations
def show_examples(original, translated):
    for orig, trans in zip(original, translated):
        print(f'Original: {orig} | Translated: {trans}')

# Save results to a file
def save_results(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f)

# Main function to run the evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate translation model.')
    parser.add_argument('--model_path', required=True, help='Path to the trained model.')
    parser.add_argument('--test_file', required=True, help='Path to the test data file.')
    parser.add_argument('--output_file', required=True, help='Path to save the evaluation results.')

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)

    # Load test data
    with open(args.test_file, 'r') as f:
        test_data = [line.strip() for line in f.readlines()]

    # Translate test data
    translations = translate(model, test_data)

    # Calculate BLEU scores
    bleu_score = calculate_bleu(test_data, translations)
    results = {'bleu_score': bleu_score}

    # Show example translations
    show_examples(test_data, translations)

    # Save results
    save_results(results, args.output_file)