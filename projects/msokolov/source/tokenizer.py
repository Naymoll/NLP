import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass
class Token:
    text: str
    span: Span
    lemma: str = ""
    stemma: str = ""


class Tokenizer:
    def __init__(self):
        times_format = r"(\d{1,2}:\d{2}(?:\s?PM|AM|pm|am)?)|(\d\s?(?:PM|AM|pm|am))"  # TODO: Доработать
        values_format = r"(\d+(?:,|\.)?\d*)"
        # punc_format = r"([^a-zA-Z\d\s])"
        punc_format = r"([\.,;!])"
        words_format = r"([a-zA-Z]+)"

        regex_formats = [
            values_format,
            punc_format,
            words_format,
        ]

        self.__regex = re.compile(r"|".join(regex_formats))

    def tokenize(self, text: str):
        result = []
        for match in self.__regex.finditer(text):
            span = Span(match.start(), match.end())
            token = Token(match.group(), span)
            result.append(token)

        return result
