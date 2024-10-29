from modules.top_selection import get_top_n_grams
import random


def generate_random_sentence(ngram_table, n, max_length=15, start_context=tuple()):

    if not start_context:
        # Noktalama içermeyen başlangıç n-gram'ları arasından seçim yap
        start_ngrams = [
            ngram for ngram in ngram_table 
            if len(ngram) == n and all(word.isalnum() for word in ngram)
        ]

        if not start_ngrams:
            raise ValueError("N-gram table does not contain any valid starting n-grams.")
        
        start_context = max(start_ngrams, key=lambda k: ngram_table[k])[:-1]

    current_context = start_context
    sentence = list(start_context)
    punctuation_limit = 3  # Noktalama işaretinden sonra minimum bu kadar kelime olmalı
    punctuation_countdown = punctuation_limit

    for _ in range(max_length - len(start_context)):
        top_ngrams = get_top_n_grams(ngram_table, current_context)

        if not top_ngrams:
            # Eğer uygun bir devam bulunamazsa yeni bir başlangıç n-gram'ı seçiyoruz
            current_context = random.choice(list(ngram_table.keys()))[:n-1]
            sentence.extend(current_context)

        else:
            # Top n-gram'lar arasından rastgele bir seçim yapıyoruz
            next_ngram = random.choices(
                top_ngrams,
                weights=[ngram_table[ngram] for ngram in top_ngrams]
            )[0]
            next_word = next_ngram[-1]

            # Noktalama işaretlerinin çok sık kullanılmasını engelle
            if next_word in {".", ","}:

                if punctuation_countdown > 0:
                    continue  # Noktalama işaretini atla
                punctuation_countdown = punctuation_limit
                
            else:
                punctuation_countdown -= 1

            # Aşırı tekrarları engelle: aynı kelime art arda 3'ten fazla tekrarlanmasın
            if len(sentence) >= 2 and sentence[-1] == sentence[-2] == next_word:
                continue  # Aynı kelime tekrarını atla

            sentence.append(next_word)
            current_context = tuple(sentence[-(n-1):])

    return ' '.join(sentence)
