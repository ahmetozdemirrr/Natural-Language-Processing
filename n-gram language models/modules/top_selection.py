import random


def get_top_n_grams(ngram_table, context, n=5):
    """
    Returns the top N n-grams that match the given context.
    :param ngram_table: Dictionary of n-grams and their frequencies.
    :param context: Tuple representing the context (previous words or characters). Can be empty.
    :param n: Number of top n-grams to return (default is 5).
    :return: List of top N n-grams.
    """
    # If context is empty, return the most frequent n-grams
    if not context:
        sorted_ngrams = sorted(ngram_table.items(), key=lambda item: item[1], reverse=True)
        return [k for k, _ in sorted_ngrams[:n]]

    # Otherwise, return the top n-grams matching the context
    candidates = {k: v for k, v in ngram_table.items() if k[:-1] == context}
    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return [k for k, _ in sorted_candidates[:n]]
