import re


vowels = "aeıioöuü"

def is_vowel(char):
    return char in vowels


def syllabify(word):
    syllables = [] # -> heceler
    current_syllable = "" # -> anlık hece

    i = 0

    while i < len(word):

        char = word[i]
        current_syllable += char

        if is_vowel(char): # sesli - ...

            if (i + 1) < len(word) and not is_vowel(word[i + 1]): # sesli - sessiz - ...

                if (i + 2) < len(word) and not is_vowel(word[i + 2]): # sesli - sessiz - sessiz - ...

                    if (i + 3) < len(word) and not is_vowel(word[i + 3]): # sesli - sessiz - sessiz - sessiz ...

                        current_syllable += word[i + 1] + word[i + 2]
                        syllables.append(current_syllable)

                        current_syllable = ""
                        i += 3
                        continue

                    else:

                        # sesli - sessiz - sessiz ise ilk iki harfi hecele

                        # eğer -> üç harfliyse (tek hecele)
                        if len(word) == 3:
                            syllables.append(word)
                            break

                        current_syllable += word[i + 1]
                        syllables.append(current_syllable)

                        current_syllable = ""
                        i += 2
                        continue
                
                else: # sesli - sessiz - sesli şeklindeyse sadece ilk harfi ekle

                    syllables.append(current_syllable)
                    current_syllable = ""
                    i += 1
                    continue

            else: # sesli - sesli - ...

                syllables.append(current_syllable)
                current_syllable = ""
                i += 1
                continue

        else: # sessiz - ...

            if (i + 1) < len(word) and is_vowel(word[i + 1]): # sessiz - sesli - ...
                
                if (i + 2) < len(word) and is_vowel(word[i + 2]): # sessiz - sesli - sesli - ...

                    if len(word) == 4:

                        # sessiz - sesli - sesli durumu -> sadece ilk iki karakter eklensin
                        current_syllable += word[i + 1]
                        syllables.append(current_syllable)
                        current_syllable = ""

                        current_syllable += word[i + 2]
                        if (i + 3) < len(word):
                            current_syllable += word[i + 3]
                        syllables.append(current_syllable)
                        current_syllable = ""
                        break

                else: # sessiz - sesli - sessiz durumu -> sonraki harfe de bakılmalı

                    if (i + 3) < len(word) and is_vowel(word[i + 3]):

                        # sessiz - sesli - sessiz - sesli durumu -> ilk ikisini al
                        current_syllable += word[i + 1]
                        syllables.append(current_syllable)

                        current_syllable = ""
                        i += 2
                        continue

                    elif (i + 3) < len(word) and not is_vowel(word[i + 3]) and (i + 4) < len(word) and not is_vowel(word[i + 4]): 

                        # sessiz - sesli - sessiz - sessiz - sessiz durumu -> ilk üç harfi hecele
                        current_syllable += word[i + 1]
                        current_syllable += word[i + 2]
                        if (i + 3) < len(word):
                            current_syllable += word[i + 3]
                        syllables.append(current_syllable)

                        current_syllable = ""
                        i += 4
                        continue

                    elif (i + 2) < len(word):  # sessiz - sesli - sessiz - sessiz durumu -> ilk ikisini al

                        current_syllable += word[i + 1]
                        current_syllable += word[i + 2]
                        syllables.append(current_syllable)

                        current_syllable = ""
                        i += 3
                        continue

            else:

                if (i + 2) < len(word) and not is_vowel(word[i + 2]): # sonraki sessiz mi?

                    # sessiz - sessiz - sessiz durumu -> kontrol => kont - rol
                    current_syllable += word[i + 1]
                    current_syllable += word[i + 2]
                    syllables.append(current_syllable)

                    current_syllable = ""
                    i += 3
                    continue
            i += 1

    # Son bir hece kalmışsa onu da ekle
    if current_syllable and len(word) != 3: # 4 harfli özel durumlar için eklenmiştir (sa-at)
        syllables.append(current_syllable)

    return syllables


def syllabify_text_with_punctuation(content):
    # Kelimeleri ve noktalama işaretlerini ayır
    words = re.findall(r'\w+|[^\w\s]', content, re.UNICODE)
    # Kelimeleri hecelerine böl, noktalama işaretlerini olduğu gibi bırak
    syllabified_words = [' '.join(syllabify(word)) if word.isalpha() else word for word in words]
    return ' '.join(syllabified_words)


if __name__ == "__main__":

    words = [   "ata",
                "saatim", 
                "saat", 
                "maaş", 
                "maaşımız", 
                "kaan", 
                "deniz", 
                "çocukluk", 
                "sallık", 
                "sallantı", 
                "altlık", 
                "korkmak", 
                "kontrol", 
                "uçurtma", 
                "antropoloji", 
                "ant", 
                "psikosomatik", 
                "nöroplastisite", 
                "triskaidekafobi", 
                "zor", 
                "kartopu", 
                "elektrokardiyografi"]

    for word in words:
        print(f"{word}: {syllabify(word)}")
