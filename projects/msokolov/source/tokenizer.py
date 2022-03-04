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
        times_format = r"((?:[0-1]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5]\d)?" \
                       r"(?:\s?(?:am|a\.m\.|AM|A\.M\.|pm|p\.m\.|PM|P\.M\.))?)" \
                       r"|" \
                       r"(?:[0-1]?[0-9]|2[0-3])" \
                       r"(?:\s?(?:am|a\.m\.|AM|A\.M\.|pm|p\.m\.|PM|P\.M\.))"
        date_format = r"(\d{1,2}(?:-|\.)\d{1,2}(?:-|\.)(?:\d{4}|\d{2}))" \
                      r"|" \
                      r"((?:\d{4}|\d{2})\-\d{1,2}-\d{1,2})"
        ip_format = r"((?:\d{1,3}\.){3}\d{1,3})"
        values_format = r"((\d+[,\.]\d+)|(\d+))"
        phone_number = r"((?:(?:\(\d{3}\))|(?:\d{3}))[\s-]\d{3}-\d{4})"
        words_format = r"([a-zA-Z]+(?:'[a-zA-z]{1,})?)"
        url = r"((?:https?:\/\/)?(?:www\.)?\w+(?:\.\w+)*\.(?:com|COM|net|NET|org|ORG)(?:\/.*?(?=\s|$))?)"
        meta = r"((&lt;.*?&gt;)|(#.*?;))"
        abbrev_format = r"((?:(?<=\s)|(?<=^))(?:[a-zA-Z]\.)+)"
        punc_format = r"(\.{1,}|[!\?,\:]|-{1,})"
        others_format = r"([^a-zA-Z\d\s]+)"

        regex_formats = [
            meta,
            url,
            ip_format,
            phone_number,
            date_format,
            times_format,
            values_format,
            abbrev_format,
            words_format,
            punc_format,
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
