import itertools

from dataclasses import dataclass
from nltk import WordNetLemmatizer, SnowballStemmer

from source.tokenizer import Tokenizer, Token


@dataclass
class Record:
    label: str
    title: str
    text: str

    def tokenize(self, tokenizer: Tokenizer):
        tokens = []

        title_tokens = tokenizer.tokenize_sentences(self.title)
        tokens.extend(title_tokens)

        text_tokens = tokenizer.tokenize_sentences(self.text)
        tokens.extend(text_tokens)

        return TokenizedRecord(int(self.label), tokens)


@dataclass
class TokenizedRecord:
    label: int
    sentences_tokens: [[Token]]

    def lemmatize(self, lemmatizer: WordNetLemmatizer):
        for token in itertools.chain.from_iterable(self.sentences_tokens):
            token.lemmatize(lemmatizer)

    def stem(self, stemmer: SnowballStemmer):
        for token in itertools.chain.from_iterable(self.sentences_tokens):
            token.stem(stemmer)
