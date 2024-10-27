import re
from tqdm import tqdm
from collections import defaultdict # 0, for default dictionary elements


# Function to generate n-grams for given text
def generate_ngrams(text, n):
    """
    Generates n-grams from the given text.
    :param text: Input text as a string.
    :param n: The n value for n-grams (1 for unigram, 2 for bigram, etc.).
    :return: List of n-grams.
    """
    tokens = text.split()
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams


# Function to build n-gram frequency tables
def build_ngram_table(data, n):
    """
    Builds n-gram frequency tables using a dictionary.
    :param data: List of strings (lines from the text file).
    :param n: The n value for n-grams (1, 2, 3, etc.).
    :return: Dictionary containing n-grams and their frequencies.
    """
    ngram_table = defaultdict(int)
    freq_count  = defaultdict(int)

    for line in tqdm(data, desc=f"Building {n}-Gram", unit="line", colour="green"):
        tokens = line.strip().split()

        # Create n-grams and update the count in the table
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngram_table[ngram] += 1

    # Count how many n-grams have each frequency
    for freq in ngram_table.values():
        freq_count[freq] += 1

    return ngram_table, freq_count


# Function to load processed data from file
def load_data(file_path):
    """
    Loads processed text data from the given file.
    :param file_path: Path to the file.
    :return: List of strings (lines from the file).
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def apply_good_turing_smoothing(ngram_table, freq_count):
    smoothed_table = {}
    N = sum(ngram_table.values())  # Total number of n-grams

    # Apply Good-Turing smoothing
    for ngram, freq in ngram_table.items():
        
        if freq + 1 in freq_count:
            smoothed_freq = (freq + 1) * freq_count[freq + 1] / freq_count[freq]
        else:
            smoothed_freq = freq  # No adjustment if there's no frequency count for freq+1
        
        smoothed_table[ngram] = smoothed_freq / N  # Normalize by total n-grams

    return smoothed_table


# Function to save n-gram tables
def save_ngram_table(ngram_table, file_path):
    """
    Saves n-gram table to a file.
    :param ngram_table: Dictionary containing n-grams and their frequencies.
    :param file_path: Path to the output file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for ngram, freq in ngram_table.items():
            ngram_str = ' '.join(ngram)
            file.write(f"{ngram_str}\t{freq}\n")


# Load n-gram table from file
def load_ngram_table(file_path):
    ngram_table = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            ngram, freq = line.strip().split('\t')
            ngram_table[tuple(ngram.split())] = float(freq)
    return ngram_table
