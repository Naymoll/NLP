import re
from dataclasses import dataclass
from source.tokenizer.patterns import *

_regex_patterns_list = [
    meta_pattern,
    url_pattern,
    ip_pattern,
    phone_pattern,
    date_pattern,
    time_pattern,
    values_pattern,
    abbrev_pattern,
    words_pattern,
    punc_pattern,
]


@dataclass
class Token:
    text: str
    lemma: str = ""
    stemma: str = ""

    def lemmatize(self, lemmatizer):
        self.lemma = lemmatizer.lemmatize(self.text)

    def stem(self, stemmer):
        self.stemma = stemmer.stem(self.text)


class Tokenizer:
    def __init__(self, regex_patterns_list=None):
        if regex_patterns_list is None:
            regex_patterns_list = _regex_patterns_list

        self.__regex = re.compile(r"|".join(regex_patterns_list))
        self.__sen_reg = re.compile(sentence_pattern)

    def tokenize(self, text: str):
        result = []
        for match in self.__regex.finditer(text):
            token = Token(match.group())
            result.append(token)

        return result

    def tokenize_sentences(self, text: str):
        result = []

        sentences = self.__sen_reg.split(text.replace('\\', ' '))
        for sentence in sentences:
            sentence_tokens = []
            for match in self.__regex.finditer(sentence):
                token = Token(match.group())
                sentence_tokens.append(token)
            result.append(sentence_tokens)

        return result
