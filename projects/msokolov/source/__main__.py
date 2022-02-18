import sys
import csv

from source.tokenizer import (Tokenizer, Token)

import nltk
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag


def init_ntlk():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')


def lemmatize(tokens: [Token]):
    lemmatizer = WordNetLemmatizer()
    result = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token.text)
        result.append(lemma)
    # tokens_text = list(map(lambda token: token.text, tokens))
    # tokens_tags = pos_tag(tokens_text, 'universal')
    # return tokens_tags
    return result


def stem(tokens: [Token]):
    stemmer = SnowballStemmer("english")
    result = []
    for token in tokens:
        stemma = stemmer.stem(token.text)
        result.append(stemma)

    return result


def read_from_file(path: str):
    text = ""
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            text = text + ', '.join(row)

    return text


def write_to_file(path: str, tokens: [Token], lemmas: [str], stemmas: [str]):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for (token, lemma, stemma) in zip(tokens, lemmas, stemmas):
            writer.writerow([token.text, lemma, stemma])


def main(path: str):
    init_ntlk()

    text = read_from_file(path)
    tokenizer = Tokenizer()

    tokens = tokenizer.tokenize(text)
    lemmas = lemmatize(tokens)
    stemmas = stem(tokens)

    write_to_file("/home/naymoll/Downloads/result.csv", tokens, lemmas, stemmas)


if __name__ == "__main__":
    main(sys.argv[1])
