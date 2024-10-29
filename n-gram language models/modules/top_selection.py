import random


def get_top_n_grams(ngram_table, current_context, top_k=5):
    """
    Returns top K n-grams that match the current context.
    :param ngram_table: N-gram table.
    :param current_context: Current context tuple.
    :param top_k: Number of top n-grams to return.
    :return: List of top n-grams.
    """
    matching_ngrams = [ngram for ngram in ngram_table if ngram[:-1] == current_context]
    
    if not matching_ngrams:
        return []
    
    # Sıralama ve en yüksek frekanslı n-gramları seçme
    matching_ngrams.sort(key=lambda x: ngram_table[x], reverse=True)
    
    return matching_ngrams[:top_k]
