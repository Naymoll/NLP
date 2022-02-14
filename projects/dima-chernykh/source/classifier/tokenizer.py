import re
from dataclasses import dataclass
from typing import List

import nltk
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

word_temp = r"([a-zA-Z]+)"
number_temp = r"([0-9]+)"
punc_temp = r"([^A-Za-z0-9\s])"

temp_list = [
    word_temp,
    number_temp,
    punc_temp
]

sum_temp = re.compile('|'.join(temp_list))


@dataclass
class Token:
    name: str
    stemma: str = None
    lemma: str = None
    tag: str = None

    def __str__(self):
        return "%s %s %s" % (self.name, self.stemma, self.lemma)

    def info(self):
        return "%s %s %s %s" % (self.name, self.stemma, self.lemma, self.tag)


def init_nltk():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')


def tokenize(text: str):
    result = []
    for tx in sum_temp.finditer(text):
        result.append(tx.group())
    return result


def tag_and_make_token_list(tokens: []):
    result = []
    for name, tag in pos_tag(tokens, tagset='universal'):
        if re.match(punc_temp, name):
            result.append(Token(name, tag=name))
        else:
            result.append(Token(name, tag=tag))
    return result


def handle_pos(tk: Token):
    pos_switcher = {
        'VERB': 'v',
        'NOUN': 'n',
        'ADJ': 'a',
        'ADV': 'r',
    }
    return pos_switcher.get(tk.tag, "n")


def handle_stemma_and_lemma(tokens: List[Token]) -> List[Token]:
    snowball = SnowballStemmer("english")
    lemma_handle = WordNetLemmatizer()
    for tk in tokens:
        stemma = snowball.stem(tk.name)
        lemma = lemma_handle.lemmatize(tk.name, pos=handle_pos(tk))
        if stemma is None:
            tk.stemma = tk.name
        tk.stemma = stemma
        tk.lemma = lemma
    return tokens


def format_tokens(text: str):
    return handle_stemma_and_lemma(tag_and_make_token_list(tokenize(text)))


# if __name__ == "__main__":
#     for tk in format_tokens("Test text"):
#         print(tk.info())
