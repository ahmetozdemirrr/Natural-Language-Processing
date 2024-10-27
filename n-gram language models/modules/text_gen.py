from modules.top_selection import get_top_n_grams
import random


def generate_random_sentence(ngram_table, start_context=tuple(), max_length=15):
    """
    Generates a random sentence using the given n-gram table.
    :param ngram_table: N-gram table.
    :param start_context: Starting context (initial words or characters).
    :param max_length: Maximum length of the generated sentence.
    :return: Generated sentence as a string.
    """

    if not start_context:
        # En sık kullanılan başlangıç n-gram’ını seç
        start_context = max(
            [ngram for ngram in ngram_table if len(ngram) == 1 or ngram[0].istitle()],
            key=lambda k: ngram_table[k]
        )

    current_context = start_context
    sentence = list(start_context)
    
    for _ in range(max_length - len(start_context)):
        # Şu anki bağlama uygun top n-gram'ları getiriyoruz
        top_ngrams = get_top_n_grams(ngram_table, current_context)

        if not top_ngrams:
            # Eğer uygun bir devam kelimesi bulunamazsa yeni bir başlangıç n-gram'ı seçiyoruz
            # print(f"...")
            current_context = random.choice(list(ngram_table.keys()))[:1]
            sentence.append(current_context[0])

        else:
            # Top n-gram'lar arasından rastgele bir seçim yapıyoruz
            next_ngram = random.choices(top_ngrams, weights=[ngram_table[ngram] for ngram in top_ngrams])[0]
            sentence.append(next_ngram[-1])  # Seçilen n-gram'ın son elemanını ekliyoruz
            current_context = tuple(sentence[-(len(current_context)):])  # Bağlamı güncelliyoruz
    
    return ' '.join(sentence)
