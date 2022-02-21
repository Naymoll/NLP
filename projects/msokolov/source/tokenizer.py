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
        times_format = r"((\d{1,2}(?::\d{1,2})? ?(?:PM|AM|pm|am|p\.m\.|a\.m\.))|(\d{1,2}:\d{2}(?:PM|AM|pm|am|p\.m\.|a\.m\.)?))" # Парсит 25:35 как корректное время
        values_format = r"((\d+[,\.]\d+)|(\d+))"
        phone_number = r"((?:(?:\(\d{3}\))|(?:\d{3}))[\s-]\d{3}-\d{4})"
        words_format = r"([a-zA-Z]+(?:'[a-zA-z]{1,})?)"
        url = r"((?:http(?:s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+)" # Работает неважно
        others_format = r"([^a-zA-Z\d\s])"

        regex_formats = [
            url,
            phone_number,
            times_format,
            values_format,
            words_format,
            others_format,
        ]

        self.__regex = re.compile(r"|".join(regex_formats))

    def tokenize(self, text: str):
        result = []
        for match in self.__regex.finditer(text):
            span = Span(match.start(), match.end())
            token = Token(match.group(), span)
            result.append(token)

        return result
