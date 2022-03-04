import re
import sys
import csv
import nltk

from nltk import SnowballStemmer, WordNetLemmatizer
from pathlib import Path
from dataclasses import dataclass

from source.tokenizer import Tokenizer, Token

abbrs = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "July",
    "Aug",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
    "Mon",
    "Mo",
    "Tue",
    "Tu",
    "Wed",
    "We",
    "Thu",
    "Th",
    "Fri",
    "Fr",
    "Sat",
    "Sa",
    "Sun",
    "Su",
    "vs",
    "VS",
    "Vs",
    "St",
    "ST",
    "Corp",
    "Inc",
    "Co",
    "Ltd",
    "Mr",
    "Mrs",
    "No",
    "Govt",
    "Gov",
    "Ark",
    "Mass",
    "Col",
    "Sgt",
    "Md",
    "Rep",
    "Rev",
    "Gen",
    "Sen",
    "Eq",
    "Dr",
    "Sep",
    "ept",
    "Tues",
    "Cos",
    "Va",
    "Wis",
    "Ga",
    "[A-Z]",
]

abbrs_neg_lookbehind_pattern = "".join(map(lambda s: rf"(?<!{s}\.)", abbrs))
sentence_reg = re.compile(rf"(?:{abbrs_neg_lookbehind_pattern})(?<=[\.!?])\s(?=[A-Z\d])")


def init_ntlk():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')


@dataclass
class Sentence:
    tokens: [Token]


@dataclass
class Record:
    label: str
    sentences: [Sentence]

    def lemmatize(self, lemmatizer: WordNetLemmatizer):
        for sentence in self.sentences:
            for token in sentence.tokens:
                token.lemma = lemmatizer.lemmatize(token.text)

    def stem(self, stemmer: SnowballStemmer):
        for sentence in self.sentences:
            for token in sentence.tokens:
                token.stemma = stemmer.stem(token.text)


def split_to_sentences(text: str):
    result = []
    tokenizer = Tokenizer()
    for sentence in sentence_reg.split(text):
        tokens = tokenizer.tokenize(sentence)
        result.append(Sentence(tokens))

    return result


def read_from_file(path: str):
    records = []
    with open(path) as file:
        reader = csv.reader(file)
        for [label, title, text] in reader:
            sentences = []
            title_sentences = split_to_sentences(title)
            sentences.extend(title_sentences)

            text_sentences = split_to_sentences(text)
            sentences.extend(text_sentences)

            records.append(Record(label, sentences))

    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    for record in records:
        record.lemmatize(lemmatizer)
        record.stem(stemmer)

    return records


def write_to_file(path: str, records: [Record]):
    last_indexes = {}
    for record in records:
        label: str = record.label

        folder = f"{path}/{label}"
        Path(folder).mkdir(parents=True, exist_ok=True)

        last_index = last_indexes.get(label, 0)
        last_indexes[label] = last_index + 1

        with open(f"{folder}/{last_index}.tsv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t', dialect='excel-tab')

            for sentence in record.sentences:
                for token in sentence.tokens:
                    writer.writerow([token.text, token.lemma, token.stemma])
                writer.writerow('')


def main(in_path: str, out_path: str):
    init_ntlk()

    records = read_from_file(in_path)
    write_to_file(out_path, records)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
