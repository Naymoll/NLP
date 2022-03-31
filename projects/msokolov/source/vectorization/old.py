import math
import re

import numpy as np
from scipy import sparse

from source.tokenizer import sentence_pattern, Tokenizer


def create_indexes(freq_dict: dict[str, int]):
    indexes = dict()
    for i, (token, _) in enumerate(freq_dict.items()):
        indexes[token] = i

    return indexes


def old_create_term_doc_matrix(text: str, _term_doc_matrix: sparse.csr_matrix, freq_dict: dict[str, int]):
    sentence_regex = re.compile(sentence_pattern)
    sentences = re.split(sentence_regex, text)

    tokenizer = Tokenizer()
    freq_dict_len = len(freq_dict)
    indexes = create_indexes(freq_dict)
    text_term_doc_matrix = sparse.lil_matrix((len(sentences), freq_dict_len), dtype=np.uint)
    for i, sentence in enumerate(sentences):
        tokens = tokenizer.tokenize(sentence)

        sentence_tokens_freq = np.zeros(freq_dict_len, dtype=np.uint)
        for token in tokens:
            token_text = token.text.lower()
            if token_text not in freq_dict:
                continue

            token_idx = indexes[token_text]
            sentence_tokens_freq[token_idx] += 1

        text_term_doc_matrix[i] = sentence_tokens_freq

    return text_term_doc_matrix.tocsr()


def count_docs_with_tokens(corpus):
    _, cols = corpus.get_shape()
    docs_count = np.zeros(cols, dtype=np.uint16)

    _, col, _ = sparse.find(corpus)
    for i in col:
        docs_count[i] += 1

    return docs_count


def count_tokens_in_docs(corpus):
    rows, _ = corpus.get_shape()
    tokens_count = np.zeros(rows, dtype=np.uint16)

    rows, _, vals = sparse.find(corpus)
    for row, val in zip(rows, vals):
        tokens_count[row] += val

    return tokens_count


def compute_idf(corpus_freq_matrix):
    docs_count = count_docs_with_tokens(corpus_freq_matrix)
    rows, cols, _ = sparse.find(corpus_freq_matrix)

    shape = corpus_freq_matrix.get_shape()
    idf_matrix = sparse.lil_matrix(shape, dtype=float)

    docs_len = shape[0]
    for row, col in zip(rows, cols):
        idf_matrix[row, col] = math.log10(docs_len / docs_count[col])

    return idf_matrix.tocsr()


def compute_tf(corpus_freq_matrix):
    tokens_count = count_tokens_in_docs(corpus_freq_matrix)
    rows, cols, vals = sparse.find(corpus_freq_matrix)

    tf_matrix = sparse.lil_matrix(corpus_freq_matrix.get_shape(), dtype=float)
    for row, col, val in zip(rows, cols, vals):
        tf_matrix[row, col] = val / tokens_count[row]

    return tf_matrix.tocsr()


def compute_tf_idf(corpus_freq_matrix):
    idf_matrix = compute_idf(corpus_freq_matrix)
    tf_matrix = compute_tf(corpus_freq_matrix)
    return idf_matrix.multiply(tf_matrix)