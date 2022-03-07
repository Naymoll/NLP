from dataclasses import dataclass
from nltk import WordNetLemmatizer, SnowballStemmer

from source.tokenizer import Token


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