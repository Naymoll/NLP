import re
from dataclasses import dataclass
from patterns import *


@dataclass
class Token:
    text: str
    lemma: str = ""
    stemma: str = ""


class Tokenizer:
    def __init__(self):
        regex_formats = [
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
            others_pattern,
        ]
        self.__regex = re.compile(r"|".join(regex_formats))

    def tokenize(self, text: str):
        result = []
        for match in self.__regex.finditer(text):
            token = Token(match.group())
            result.append(token)

        return result
