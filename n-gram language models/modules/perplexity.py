import math
from collections import defaultdict


# Load n-gram model probabilities
def load_ngram_probabilities(file_path):
    ngram_probs = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            ngram, prob = line.strip().split('\t')
            ngram_probs[tuple(ngram.split())] = float(prob)
    return ngram_probs


# Calculate perplexity for the given test data and n-gram model
def calculate_perplexity(test_data, ngram_probs, n):
    total_log_prob = 0
    total_ngrams = 0
    
    for line in test_data:
        tokens = line.strip().split()
        
        # Iterate over each n-gram in the line
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])

            if ngram in ngram_probs:
                total_log_prob += math.log(ngram_probs[ngram])
            else:
                # If n-gram not found, assign a very small probability
                total_log_prob += math.log(1e-10)  
            total_ngrams += 1

    # Calculate perplexity
    avg_log_prob = total_log_prob / total_ngrams
    perplexity = math.exp(-avg_log_prob)
    
    return perplexity


if __name__ == "__main__":
    # Example usage:
    n = 2  # for bigram model
    test_file_path = "./data/processed/wiki_00_syllables_test.txt"
    ngram_prob_file = "./results/syllable_ngram_2-gram.txt"

    # Load test data and n-gram probabilities
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    ngram_probs = load_ngram_probabilities(ngram_prob_file)
    
    # Calculate perplexity
    perplexity = calculate_perplexity(test_data, ngram_probs, n)
    
    print(f"Perplexity for {n}-gram model: {perplexity}")