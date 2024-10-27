import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_preperation import process_text_file
from modules.split_data import split_data
from modules.ngram_calculation import build_ngram_table, load_data, save_ngram_table, apply_good_turing_smoothing, load_ngram_table
from modules.perplexity import calculate_perplexity
from modules.text_gen import generate_random_sentence
from modules.tables import save_perplexity_table, save_sample_sentences_table

def check_processed_data_exists():
    return os.path.exists('./data/processed/wiki_00_syllables.txt') and os.path.exists('./data/processed/wiki_00_characters.txt')


def calculate_and_save_ngrams(file_path, output_prefix, n):
    """
    Calculates n-gram tables for given data and saves them.
    :param file_path: Path to the input file.
    :param output_prefix: Prefix for output file path.
    :param n: N-gram value (1 for unigram, 2 for bigram, etc.).
    """
    data = load_data(file_path)
    
    # Build n-gram table and frequency count
    ngram_table, freq_count = build_ngram_table(data, n)
    # Apply Good-Turing smoothing
    smoothed_table = apply_good_turing_smoothing(ngram_table, freq_count)
    # Output path for the n-gram table
    output_path = f"{output_prefix}{n}-gram.txt"
    # Save the smoothed n-gram table
    save_ngram_table(smoothed_table, output_path)

    return smoothed_table


def main():
    parser = argparse.ArgumentParser(description="NLP Pipeline")
    parser.add_argument('--clean', action='store_true', help="Run data cleaning and preprocessing.")
    parser.add_argument('--ngram', action='store_true', help="Run n-gram model generation.")
    parser.add_argument('--perplexity', action='store_true', help="Calculate and display perplexity")
    parser.add_argument('--textgen', action='store_true', help="Generate random sentences using n-gram models.")

    args = parser.parse_args()

    if args.clean:

        print("\033[33m<-> Running data cleaning and preprocessing...\033[0m")
        raw_file = "./data/raw/wiki_00.txt"
        
        syllable_output = "./data/processed/wiki_00_syllables.txt"
        character_output = "./data/processed/wiki_00_characters.txt"
        
        # Veri temizleme ve ayırma işlemi
        process_text_file(raw_file, syllable_output, model_type="syllable")
        process_text_file(raw_file, character_output, model_type="character")
        # Veriyi eğitim ve test olarak ayırma
        print("\033[33m<-> Data extraction process begins")
        split_data(syllable_output, "./data/processed/wiki_00_syllables_train.txt", "./data/processed/wiki_00_syllables_test.txt")
        split_data(character_output, "./data/processed/wiki_00_characters_train.txt", "./data/processed/wiki_00_characters_test.txt")

        print("\033[33m<-> Data cleaning and preprocessing completed.\033[0m")


    if args.ngram:

        if not check_processed_data_exists():

            print("\033[33m<-> Processed data not found. Please run with --clean first.\033[0m")
            return

        print("\033[33m<-> Running n-gram model generation...\033[0m")
        
        # File paths
        syllable_train_file = "./data/processed/wiki_00_syllables_train.txt"
        character_train_file = "./data/processed/wiki_00_characters_train.txt"
        # Output paths
        syllable_ngram_output = "./results/syllable_ngram_"
        character_ngram_output = "./results/character_ngram_"
        
        # Store n-gram tables in dictionaries
        syllable_ngram_tables = {}
        character_ngram_tables = {}

         # Generate and save n-grams for syllable-based data
        for n in range(1, 4):
            syllable_ngram_tables[n] = calculate_and_save_ngrams(syllable_train_file, syllable_ngram_output, n)

        # Generate and save n-grams for character-based data
        for n in range(1, 4):
            character_ngram_tables[n] = calculate_and_save_ngrams(character_train_file, character_ngram_output, n)

        print("\033[33m<-> N-gram model generation completed.\033[0m")


    if args.perplexity:

        # Load n-gram tables from files if needed
        syllable_ngram_tables = {}
        character_ngram_tables = {}

        for n in range(1, 4):
            syllable_ngram_tables[n] = load_ngram_table(f"./results/syllable_ngram_{n}-gram.txt")
            character_ngram_tables[n] = load_ngram_table(f"./results/character_ngram_{n}-gram.txt")

        # Calculate perplexity with data and n-gram tables used in education
        syllable_test_file  = "./data/processed/wiki_00_syllables_test.txt"
        character_test_file = "./data/processed/wiki_00_characters_test.txt"

        syllable_test_data  = load_data(syllable_test_file)
        character_test_data = load_data(character_test_file)

        perplexity_data = []

        print("\033[33m<-> Calculating perplexity for syllable-based model...\033[0m")
        for n in range(1, 4):
            perplexity = calculate_perplexity(syllable_test_data, syllable_ngram_tables[n], n)
            print(f"\033[33mSyllable-based {n}-gram perplexity:\033[0m {perplexity}")
            perplexity_data.append(("Syllable-Based", f"{n}-Gram", perplexity))

        print("\033[33m<-> Calculating perplexity for character-based model...\033[0m")
        for n in range(1, 4):
            perplexity = calculate_perplexity(character_test_data, character_ngram_tables[n], n)
            print(f"\033[33mCharacter-based {n}-gram perplexity:\033[0m {perplexity}")
            perplexity_data.append(("Character-Based", f"{n}-Gram", perplexity))

        # Save the perplexity table to a .csv file
        save_perplexity_table(perplexity_data, filename="perplexity_results.csv")
        print("\033[32m<-> Perplexity Values Table is saved at: perplexity_results.csv\033[0m")


    if args.textgen:
        # Load n-gram models for sentence generation
        syllable_ngram_table = {n: load_ngram_table(f"./results/syllable_ngram_{n}-gram.txt") for n in range(1, 4)}
        character_ngram_table = {n: load_ngram_table(f"./results/character_ngram_{n}-gram.txt") for n in range(1, 4)}

        sentences_data = []

        print("\033[33m<-> Generating random sentences for syllable-based model...\033[0m")
        for n in range(1, 4):
            sentence1 = generate_random_sentence(syllable_ngram_table[n], max_length=50)
            sentence2 = generate_random_sentence(syllable_ngram_table[n], max_length=50)
            print(f"\033[33mSyllable-based {n}-gram sentence:\033[0m {sentence1}")
            sentences_data.append(("Syllable-Based", f"{n}-Gram", sentence1, sentence2))

        print("\033[33m<-> Generating random sentences for character-based model...\033[0m")
        for n in range(1, 4):
            sentence1 = generate_random_sentence(character_ngram_table[n], max_length=50)
            sentence2 = generate_random_sentence(character_ngram_table[n], max_length=50)
            print(f"\033[33mCharacter-based {n}-gram sentence:\033[0m {sentence1}")
            sentences_data.append(("Character-Based", f"{n}-Gram", sentence1, sentence2))

        # Save the sample sentences table to a .csv file
        save_sample_sentences_table(sentences_data, filename="sample_sentences.csv")
        print("\033[32m<-> Sample Sentences Table is saved at: sample_sentences.csv\033[0m")


if __name__ == "__main__":
    main()
