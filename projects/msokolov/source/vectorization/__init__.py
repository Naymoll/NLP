import csv
import itertools
import math

import nltk
import numpy as np
from scipy import sparse

from source.tokenizer import Tokenizer
from source.tokenizer.patterns import *
from util import read_forbidden_words


class Vocab:
    def __init__(self, vocab: dict[str, int], indexes: [dict, int]):
        self.__vocab = vocab
        self.__indexes = indexes
        pass

    @classmethod
    def from_corpus(cls, corpus: [[str]], forbidden_words: set, min_count: int = 5):
        vocab: dict[str, int] = {}
        for word in itertools.chain.from_iterable(corpus):
            if not word or word in forbidden_words:
                continue

            vocab[word] = vocab.get(word, 0) + 1  # If not in dict insert 1

        index = 0
        result_vocab: dict[str, int] = {}
        result_indexes: dict[str, int] = {}
        for word, freq in vocab.items():
            if freq < min_count:
                continue

            result_vocab[word] = freq
            result_indexes[word] = index
            index += 1

        return cls(result_vocab, result_indexes)

    def filter_corpus(self, corpus: [[str]]):
        filtred = []
        for doc in corpus:
            doc_words = []
            for word in doc:
                if word not in self.__vocab:
                    continue
                doc_words.append(word)
            filtred.append(doc_words)

        return filtred

    def save(self, path: str):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            for token, freq in self.__vocab.items():
                writer.writerow([token, freq])

    @classmethod
    def load(cls, path: str):
        with open(path) as file:
            vocab = {}
            reader = csv.reader(file)
            for [token, freq] in reader:
                vocab[token] = freq

            indexes = {}
            for i, (word, _) in enumerate(vocab.items()):
                indexes[word] = i

            return cls(vocab, indexes)

    def index(self, item: str):
        return self.__indexes[item]

    def items(self):
        return self.__vocab.items()

    def __getitem__(self, item: str):
        return self.__vocab[item]

    def __len__(self):
        return len(self.__vocab)

    def __contains__(self, item):
        return self.__vocab.__contains__(item)


class TDMatrix:
    def __init__(self, td_matrix: sparse.csr_matrix, idf: np.array, vocab: Vocab):
        self.td_matrix = td_matrix
        self.idf = idf
        self.vocab = vocab

    @classmethod
    def from_corpus(cls, corpus: [[str]], vocab: Vocab = None):
        if vocab is None:
            forbidden_words = read_forbidden_words("../../assets/annotated-corpus/forbidden-words.csv")
            vocab = Vocab.from_corpus(corpus, forbidden_words)

        shape = (len(corpus), len(vocab))
        td_matrix = sparse.lil_matrix(shape, dtype=np.uint)
        for i, doc in enumerate(corpus):
            td_row = np.zeros(shape[1], dtype=np.uint)
            for word in doc:
                if word not in vocab:
                    continue

                idx = vocab.index(word)
                td_row[idx] += 1

            td_matrix[i] = td_row

        td_matrix = td_matrix.tocsr()
        idf = cls.__precompute_idf(td_matrix)
        return cls(td_matrix, idf, vocab)

    def save(self, folder_path: str):
        vocab_path = f"{folder_path}/vocab.csv"
        td_path = f"{folder_path}/td.npz"
        idf_path = f"{folder_path}/idf.npy"

        sparse.save_npz(td_path, self.td_matrix)
        np.save(idf_path, self.idf)
        self.vocab.save(vocab_path)

    @classmethod
    def load(cls, folder_path: str):
        vocab_path = f"{folder_path}/vocab.csv"
        td_path = f"{folder_path}/td.npz"
        idf_path = f"{folder_path}/idf.npy"

        td_matrix = sparse.load_npz(td_path)
        idf = np.load(idf_path)
        vocab = Vocab.load(vocab_path)

        return cls(td_matrix, idf, vocab)

    @staticmethod
    def __precompute_idf(td_matrix: sparse.csr_matrix):
        rows, cols = td_matrix.get_shape()
        counts = np.zeros(cols, dtype=np.uint)

        _, cols, _ = sparse.find(td_matrix)
        for col in cols:
            counts[col] += 1

        result = np.array(counts, dtype=float)
        for i, count in enumerate(counts):
            if count != 0:
                result[i] = math.log10(rows / count)

        return result


class TextVectorizer:
    def __init__(self, td_matrix: TDMatrix):
        patterns = [
            words_pattern,
            abbrev_pattern
        ]
        self.__tokenizer = Tokenizer(patterns)
        self.__lemmatizer = nltk.WordNetLemmatizer()
        self.__td_matrix = td_matrix

    def vectorize(self, text: str):
        sentences = self.__tokenizer.tokenize_sentences(text)

        corpus = []
        for sentence in sentences:
            doc = []
            for token in sentence:
                token.lemmatize(self.__lemmatizer)
                lemma = token.lemma.lower()

                if lemma not in self.__td_matrix.vocab:
                    continue

                doc.append(lemma)

            corpus.append(doc)

        td_matrix = TDMatrix.from_corpus(corpus, self.__td_matrix.vocab)
        tf_matrix = TextVectorizer.__create_tf_matrix(td_matrix.td_matrix)
        idf_matrix = TextVectorizer.__create_idf_matrix(td_matrix.td_matrix, self.__td_matrix.idf)
        tf_idf_matrix = tf_matrix.multiply(idf_matrix)

        return TextVectorizer.__vectorize(tf_idf_matrix)

    @staticmethod
    def __vectorize(tf_idf_matrix: sparse.csr_matrix):
        _, cols = tf_idf_matrix.get_shape()
        sums = np.zeros(cols, dtype=float)
        counts = np.zeros(cols, dtype=int)

        _, cols, vals = sparse.find(tf_idf_matrix)
        for col, val in zip(cols, vals):
            sums[col] += val
            counts[col] += 1

        result_vector = np.zeros(len(sums), dtype=float)
        np.divide(sums, counts, out=result_vector, where=counts != 0)

        return result_vector

    @staticmethod
    def __create_tf_matrix(td_matrix: sparse.csr_matrix):
        _, _, vals = sparse.find(td_matrix)
        total = np.sum(vals)

        tfs = sparse.lil_matrix(td_matrix, dtype=float)
        rows, cols, vals = sparse.find(tfs)
        for row, col, val in zip(rows, cols, vals):
            tfs[row, col] = val / total

        return tfs.tocsr()

    @staticmethod
    def __create_idf_matrix(td_matrix: sparse.csr_matrix, idf: np.array):
        corpus_idfs = sparse.lil_matrix(td_matrix, dtype=float)
        rows, cols, vals = sparse.find(corpus_idfs)
        for row, col, val in zip(rows, cols, vals):
            corpus_idfs[row, col] = idf[col]

        return corpus_idfs.tocsr()
