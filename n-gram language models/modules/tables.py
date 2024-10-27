import pandas as pd


def save_perplexity_table(perplexity_data, filename="perplexity_results.csv"):
    """
    Saves Perplexity data as a table in a .csv file.
    :param perplexity_data: List of perplexity values, e.g. [(model, ngram, perplexity), ...]
    :param filename: Name of the file to save
    """
    df = pd.DataFrame(perplexity_data, columns=["Model", "N-Gram", "Perplexity"])
    df.to_csv(filename, index=False)


def save_sample_sentences_table(sentences_data, filename="sample_sentences.csv"):
    """
    Saves sample sentences as a table in a .csv file.
    :param sentences_data: List containing sentence data, e.g. [(model, ngram, sentence1, sentence2), ...]
    :param filename: Name of the file to save
    """
    df = pd.DataFrame(sentences_data, columns=["Model", "N-Gram", "Sentence 1", "Sentence 2"])
    df.to_csv(filename, index=False)

