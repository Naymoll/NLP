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
        times_format_full = r"((?:[0-1]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5]\d)?(?:\s?(?:am|a\.m\.|AM|A\.M\.|pm|p\.m\.|PM|P\.M\.))?)"
        times_format_short = r"((?:[0-1]?[0-9]|2[0-3])(?:\s?(?:am|a\.m\.|AM|A\.M\.|pm|p\.m\.|PM|P\.M\.)))"
        values_format = r"((\d+[,\.]\d+)|(\d+))"
        phone_number = r"((?:(?:\(\d{3}\))|(?:\d{3}))[\s-]\d{3}-\d{4})"
        words_format = r"([a-zA-Z]+(?:'[a-zA-z]{1,})?)"
        url = r"((?:https?:\/\/)?(?:www\.)?\w+(?:\.\w+)*\.(?:com|COM|net|NET|org|ORG)(?:\/.*?(?=\s))?)"
        meta = r"(&lt;.*?&gt;)|(#.*?;)"
        others_format = r"([^a-zA-Z\d\s])"

        regex_formats = [
            meta,
            url,
            phone_number,
            times_format_full,
            times_format_short,
            values_format,
            words_format,
            others_format,
        ]

        self.__regex = re.compile(r"|".join(regex_formats))

    def tokenize(self, text: str):
        result = []
        for match in self.__regex.finditer(text):
            if match.group(1) or match.group(2):
                continue  # Игнорируем мета-токены

            span = Span(match.start(), match.end())
            token = Token(match.group(), span)
            result.append(token)

        return result
