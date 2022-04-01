import csv
import os

forbidden_words = set()


def read_forbidden_words(path: str):
    with open(path) as file:
        reader = csv.reader(file)
        for record in reader:
            word = record[0]
            if word not in forbidden_words:
                forbidden_words.add(word)

    return forbidden_words


def read_directory(path: str):
    result = []
    for file in os.listdir(path):
        abs_path = os.path.join(path, file)
        if not os.path.isdir(abs_path):
            result.append(abs_path)
        else:
            file_paths = read_directory(abs_path)
            result.extend(file_paths)

    return result


def read_token_file(path: str):
    tokens = []
    with open(path) as file:
        reader = csv.reader(file, delimiter='\t')
        for record in reader:
            if not record:
                continue

            token = record[0].lower()
            if token in forbidden_words:
                continue

            tokens.append(token)

    return tokens


def read_token_files(file_paths: [str]):
    freq_dict = {}
    corpus = []
    for file_path in file_paths:
        doc = read_token_file(file_path)
        for token in doc:
            freq_dict[token] = freq_dict.get(token, 0) + 1  # If not in dict insert 1

        corpus.append(doc)

    return corpus, freq_dict