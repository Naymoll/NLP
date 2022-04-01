import sys
import csv
import nltk
import itertools

from nltk import SnowballStemmer, WordNetLemmatizer
from pathlib import Path

from source.tokenizer import Tokenizer
from record import Record, TokenizedRecord


def init_ntlk():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')


def read_records(path: str):
    records = []
    with open(path) as file:
        reader = csv.reader(file)
        for [label, title, text] in reader:
            record = Record(label, title, text)
            records.append(record)

    return records


def tokenize_records(records: [Record]):
    tokenizer = Tokenizer()
    tokenized_records = []
    for record in records:
        tokenized_record = record.tokenize(tokenizer)
        tokenized_records.append(tokenized_record)

    return tokenized_records


def lemmatize_and_stem(records: [TokenizedRecord]):
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    for record in records:
        record.lemmatize(lemmatizer)
        record.stem(stemmer)


def write_records(path: str, records: [TokenizedRecord]):
    last_indexes = {}
    for record in records:
        label: str = record.label

        folder = f"{path}/{label}"
        Path(folder).mkdir(parents=True, exist_ok=True)

        last_index = last_indexes.get(label, 0)
        last_indexes[label] = last_index + 1

        with open(f"{folder}/{last_index}.tsv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t', dialect='excel-tab')

            for sentence_tokens in record.sentences_tokens:
                for token in sentence_tokens:
                    writer.writerow([token.text, token.lemma, token.stemma])
                writer.writerow('')


def create_dict(records: [TokenizedRecord]):
    tokens = set()
    for record in records:
        for token in itertools.chain.from_iterable(record.sentences_tokens):
            if token.text not in tokens:
                tokens.add(token.text)

    with open(r"../../assets/dictionary.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda t: [t], tokens))


def main(in_path: str, out_path: str):
    init_ntlk()

    records = read_records(in_path)
    tokenized_records = tokenize_records(records)
    lemmatize_and_stem(tokenized_records)
    create_dict(tokenized_records)
    write_records(out_path, tokenized_records)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
