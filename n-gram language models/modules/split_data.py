from sklearn.model_selection import train_test_split


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def write_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(data)
    print(f"\033[33m<-> File saved at: {file_path}\033[0m")


def split_data(file_path, train_file_path, test_file_path, test_size=0.05):
    data = read_file(file_path)
    # Veriyi %95 eğitim, %5 test olacak şekilde ayır
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    write_file(train_data, train_file_path)
    write_file(test_data, test_file_path)


if __name__ == "__main__":
    syllable_file = "./data/processed/wiki_00_syllables.txt"
    split_data(syllable_file, "./data/processed/wiki_00_syllables_train.txt", "./data/processed/wiki_00_syllables_test.txt")
    character_file = "./data/processed/wiki_00_characters.txt"
    split_data(character_file, "./data/processed/wiki_00_characters_train.txt", "./data/processed/wiki_00_characters_test.txt")

    print("The data was split into 95% training and 5% testing.")



