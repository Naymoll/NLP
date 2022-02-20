import sys
import csv
import nltk

from nltk import SnowballStemmer, WordNetLemmatizer
from pathlib import Path
from dataclasses import dataclass, field

from source.tokenizer import Tokenizer, Token


@dataclass
class Record:
    label: str
    title_tokens: [Token] = field(default_factory=list)
    text_tokens: [Token] = field(default_factory=list)

    def lemmatize(self, lemmatizer: WordNetLemmatizer):
        for token in self.title_tokens:
            token.lemma = lemmatizer.lemmatize(token.text)

        for token in self.text_tokens:
            token.lemma = lemmatizer.lemmatize(token.text)

    def stem(self, stemmer: SnowballStemmer):
        for token in self.title_tokens:
            token.stemma = stemmer.stem(token.text)

        for token in self.text_tokens:
            token.stemma = stemmer.stem(token.text)


def init_ntlk():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')


def lemmatize(records: [Record]):
    lemmatizer = WordNetLemmatizer()
    for record in records:
        record.lemmatize(lemmatizer)


def stem(records: [Record]):
    stemmer = SnowballStemmer("english")
    for record in records:
        record.stem(stemmer)


def read_from_file(path: str):
    tokenizer = Tokenizer()
    result = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            label = row[0]
            title_tokens = tokenizer.tokenize(row[1])
            text_tokens = tokenizer.tokenize(row[2])
            result.append(Record(label, title_tokens, text_tokens))

    return result


def write_to_file(path: str, records: [Record]):
    last_indexes = {}
    for record in records:
        label: str = record.label

        folder = f"{path}/{label}"
        Path(folder).mkdir(parents=True, exist_ok=True)

        last_index = last_indexes.get(label, 0)
        last_indexes[label] = last_index + 1

        with open(f"{folder}/{last_index}.tsv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')

            for token in record.title_tokens:
                writer.writerow([token.text, token.lemma, token.stemma])

            for token in record.text_tokens:
                writer.writerow([token.text, token.lemma, token.stemma])


def main(in_path: str, out_path: str):
    init_ntlk()

    records = read_from_file(in_path)
    lemmatize(records)
    stem(records)
    write_to_file(out_path, records)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
