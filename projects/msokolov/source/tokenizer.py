import re
from dataclasses import dataclass
from enum import Enum


class Tag(Enum):
    Undefined = 0
    Value = 1
    Punc = 2
    Word = 3


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Token:
    text: str
    span: Span
    tag: Tag


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
            tag = Tag.Undefined
            match match.groups():
                case (_, None, None):
                    tag = Tag.Value
                case (None, _, None):
                    tag = Tag.Punc
                case (None, None, _):
                    tag = Tag.Word

            span = Span(match.start(), match.end())
            token = Token(match.group(), span, tag)
            result.append(token)

        return result
