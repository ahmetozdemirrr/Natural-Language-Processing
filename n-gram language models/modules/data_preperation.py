import os
import re
import inflect
from tqdm import tqdm
from turkish_syllable import syllabify
from sklearn.model_selection import train_test_split


turkish_char_map = {
    'ü': 'u', 'Ü': 'U', 'ö': 'o', 'Ö': 'O', 'ı': 'i', 'İ': 'I',
    'ş': 's', 'Ş': 'S', 'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G',
    'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u', 'W': 'w',
    'Q': 'q', 'X': 'x'
}

html_tags = [
    "a", "abbr", "address", "area", "article", "aside", 
    "audio", "b", "base", "bdi", "bdo", "blockquote", "body",
    "br", "button", "canvas", "caption", "cite", "code", 
    "col", "colgroup", "data", "datalist", "dd", "del", 
    "details", "dfn", "dialog", "div", "dl", "dt", "em", 
    "embed", "fieldset", "figcaption", "figure", "footer", 
    "form", "h1", "h2", "h3", "h4", "h5", "h6", "head", 
    "header", "hr", "html", "i", "iframe", "img", "input", 
    "ins", "kbd", "label", "legend", "li", "link", "main", 
    "map", "mark", "meta", "meter", "nav", "noscript", "object", 
    "ol", "optgroup", "option", "output", "p", "param", 
    "picture", "pre", "progress", "q", "rp", "rt", "ruby", 
    "s", "samp", "script", "section", "select", "small",
    "source", "span", "strong", "style", "sub", "summary", 
    "sup", "table", "tbody", "td", "template", "textarea", 
    "tfoot", "th", "thead", "time", "title", "tr", "track", 
    "u", "ul", "var", "video", "wbr"
]

abbreviation_map = {
    "örn.": "örneğin",
    "vs.": "vesaire",
    "bkz.": "bakınız",
    "sn.": "sayın",
    "dr.": "doktor",
    "müh.": "mühendis",
    "prof.": "profesör",
    "av.": "avukat",
    "cad.": "cadde",
    "sok.": "sokak",
    "hz.": "hazreti",
    "m.ö.": "milattan önce",
    "m.s.": "milattan sonra",
    "km/s": "kilometre saat",
    "m/s": "metre saniye",
}

def remove_non_turkish_characters(content):
    # Sadece Türkçe alfabesi ve yaygın işaretlerin kalmasını sağla
    allowed_chars = re.compile(r'[a-zA-ZçÇğĞıİöÖşŞüÜqQwWxX\s.,!?\"\':;()/]')
    return ''.join(filter(allowed_chars.match, content))


def remove_non_turkish_words(content):
    # Sadece Türkçe karakterlerden oluşan kelimeleri filtrele
    allowed_word_pattern = re.compile(r'^[abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ.,!?]+$')
    words = content.split()
    # Türkçe dışı karakter barındıran kelimeleri ayıkla
    filtered_words = [word for word in words if allowed_word_pattern.match(word)]
    return ' '.join(filtered_words)


# Türkçe karakterleri düz Latin harflerine çevir
def replace_turkish_characters(content):
    for char, replacement in turkish_char_map.items():
        content = content.replace(char, replacement)

    return content


def turkish_lower(text):
    translation_table = str.maketrans(
        "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
        "abcçdefgğhıijklmnoöprsştuüvyz"
    )
    return text.translate(translation_table)


# 1. HTML ve <doc> etiketlerini temizleme fonksiyonu
def clean_html_tags(content):
    # Hem HTML etiketlerini hem de Wikipedia Dump etiketlerini yakalayan desen
    pattern = re.compile(r'<(/?)(\w+)\s*([^>]*)>')
    
    def replace_tag(match):

        tag = match.group(2)

        # HTML ya da Wikipedia Dump etiketi mi?
        if tag.lower() in html_tags or tag.lower() == "doc":
            return ""  # Evetse, etiketi kaldır
        else:
            return match.group(0)  # Eğer etiket değilse, dokunma
    
    cleaned_text = pattern.sub(replace_tag, content)

    return cleaned_text


# 2. Türkçe sayıları kelimelere çeviren fonksiyon
def convert_number_to_turkish(number):
    units = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
    tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
    hundreds = ["", "yüz"]
    
    if number == 0:
        return "sıfır"

    words = []

    if number >= 1000:
        thousands = number // 1000
        if thousands > 1:
            words.append(convert_number_to_turkish(thousands))  # Binler basamağını tekrar çağır
        words.append("bin")
        number %= 1000  # Binler basamağını çıkar

    if number >= 100:
        hundreds_digit = number // 100
        if hundreds_digit > 1:
            words.append(units[hundreds_digit])
        words.append(hundreds[1])  # 'yüz' eklenir
        number %= 100  # Yüzler basamağını çıkar

    if number >= 10:
        tens_digit = number // 10
        words.append(tens[tens_digit])
        number %= 10  # Onlar basamağını çıkar

    if number > 0:
        words.append(units[number])  # Birler basamağı ekle

    return " ".join(words)


def convert_num_to_word(content):

    def replace_numbers(match):
        num = int(match.group(0))
        return convert_number_to_turkish(num)  # Sayıları Türkçeye çevir
    
    return re.sub(r'\b\d+\b', replace_numbers, content)


# 3. Büyük harfleri küçüğe çevirme
def to_lower_case(content):
    return content.lower()


def remove_special_spaces(content):
    # 0xA0 (non-breaking space) karakterini normal boşlukla değiştir
    return content.replace('\xa0', ' ')


# Linkleri kaldıran fonksiyon
def remove_links(content):
    link_pattern = re.compile(r'(https?://[^\s]+|www\.[^\s]+)')
    cleaned_content = link_pattern.sub('', content)  # Linkleri boş string ile değiştir
    return cleaned_content


def expand_abbreviations(content):
    for abbr, full in abbreviation_map.items():
        content = content.replace(abbr, full)
    return content


# Heceleme ve noktalama işaretlerini ayırma
def syllabify_text_with_punctuation(content):
    words = re.findall(r'\w+|[^\w\s]', content, re.UNICODE)
    syllabified_words = [' '.join(syllabify(word)) if word.isalpha() else word for word in words]
    return ' '.join(syllabified_words)


# Karakter bazlı ayrıştırma
def char_based_text(content):
    return ' '.join(list(content))


def process_text(content, model_type="syllable"):
    content = remove_links(content)
    content = remove_special_spaces(content)
    content = clean_html_tags(content)
    # content = remove_non_turkish_characters(content)
    content = remove_non_turkish_words(content)
    content = expand_abbreviations(content)
    content = convert_num_to_word(content)
    # content = replace_turkish_characters(content)
    content = turkish_lower(content)

    if model_type == "syllable":
        content = syllabify_text_with_punctuation(content)

    elif model_type == "character":
        content = char_based_text(content)

    return content


# Dosya işleme fonksiyonu (heceleme ve karakter bazlı ayrıştırma için ayrı dosyalar)
def process_text_file(file_path, output_file_path, model_type="syllable"):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Dosya satır sayısını öğrenmek için
    with open(file_path, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)  # Toplam satır sayısı

    title = "Processing syllable  " if model_type == "syllable" else "Processing character "

    with open(file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        # tqdm ile ilerleme çubuğu
        for line in tqdm(infile, total=total_lines, desc=title, unit="line", colour="blue"):
            processed_line = process_text(line, model_type=model_type)
            outfile.write(processed_line + "\n")

    return output_file_path


if __name__ == "__main__":
    test_file = "./data/raw/wiki_00.txt"
    
    syllable_output_file = "./data/processed/wiki_00_syllables.txt"
    process_text_file(test_file, syllable_output_file, model_type="syllable")
    character_output_file = "./data/processed/wiki_00_characters.txt"
    process_text_file(test_file, character_output_file, model_type="character")

    print(f"Syllable processed file saved at: {syllable_output_file}")
    print(f"Character processed file saved at: {character_output_file}")
