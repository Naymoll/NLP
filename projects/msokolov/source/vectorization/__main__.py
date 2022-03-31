import math
import os
import re
import sys
import csv
import numpy as np
import gensim

from source.tokenizer.patterns import sentence_pattern
from source.tokenizer import Tokenizer
from source.vectorization.old import create_indexes
from source.vectorization.util import read_forbidden_words, read_directory, read_token_files

from scipy import sparse
from scipy.spatial.distance import cosine

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

FILE_LIMIT = 1000


def create_term_doc_matrix(corpus: [[str]], freq_dict: dict[str, int]):
    indexes = create_indexes(freq_dict)
    term_doc_matrix = sparse.lil_matrix((len(corpus), len(freq_dict)), dtype=np.uint)
    for doc_idx, doc in enumerate(corpus):
        doc_token_freq = np.zeros(len(freq_dict), dtype=np.uint)
        for token in doc:
            if token not in freq_dict:
                continue

            token_idx = indexes[token]
            doc_token_freq[token_idx] += 1

        term_doc_matrix[doc_idx] = doc_token_freq

    return term_doc_matrix.tocsr()


def save_freq_dict(freq_dict, path: str):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for token, freq in freq_dict.items():
            writer.writerow([token, freq])


def read_freq_dict(path: str):
    freq_dict = {}
    with open(path) as file:
        reader = csv.reader(file)
        for [token, freq] in reader:
            freq_dict[token] = freq

    return freq_dict


def init_load(freq_file_path: str, term_doc_file_path: str, min_count=2):
    if os.path.isfile(freq_file_path) and os.path.isfile(term_doc_file_path):
        return read_freq_dict(freq_file_path), sparse.load_npz(term_doc_file_path)

    read_forbidden_words("../../assets/annotated-corpus/forbidden-words.csv")
    file_paths = read_directory("../../assets/train")
    corpus, freq_dict = read_token_files(file_paths)
    model = gensim.models.Word2Vec(corpus, vector_size=10, min_count=2)
    model.save("../../assets/w2v-data.txt")

    freq_dict = dict(filter(lambda entry: entry[1] >= min_count, freq_dict.items()))  # Filter dict if freq < min_count
    term_doc_matrix = create_term_doc_matrix(corpus, freq_dict)

    save_freq_dict(freq_dict, freq_file_path)
    sparse.save_npz(term_doc_file_path, term_doc_matrix)

    return freq_dict, term_doc_matrix


def create_tf_matrix(corpus_term_doc_matrix: sparse.csr_matrix):
    _, _, vals = sparse.find(corpus_term_doc_matrix)
    total = np.sum(vals)

    tfs = sparse.lil_matrix(corpus_term_doc_matrix, dtype=float)
    rows, cols, vals = sparse.find(tfs)
    for row, col, val in zip(rows, cols, vals):
        tfs[row, col] = val / total

    return tfs.tocsr()


def create_idf_matrix(corpus_term_doc_matrix: sparse.csr_matrix, term_doc_idfs: np.array):
    corpus_idfs = sparse.lil_matrix(corpus_term_doc_matrix, dtype=float)
    rows, cols, vals = sparse.find(corpus_idfs)
    for row, col, val in zip(rows, cols, vals):
        corpus_idfs[row, col] = term_doc_idfs[col]

    return corpus_idfs.tocsr()


def get_matrices(text: str, term_doc_idf: np.array, freq_dict: dict[str, int]):
    sentence_regex = re.compile(sentence_pattern)
    sentences = re.split(sentence_regex, text)

    tokenizer = Tokenizer()
    corpus = []

    for sentence in sentences:
        tokens_in_sent = tokenizer.tokenize(sentence)
        tokens: list[str] = list(map(lambda token: token.text.lower(), tokens_in_sent))
        corpus.append(tokens)

    corpus_term_doc_matrix = create_term_doc_matrix(corpus, freq_dict)
    corpus_tf = create_tf_matrix(corpus_term_doc_matrix)
    corpus_idf = create_idf_matrix(corpus_term_doc_matrix, term_doc_idf)

    return corpus_term_doc_matrix, corpus_tf.multiply(corpus_idf)


def vectorize_tf_idf_matrix(tf_idf_matrix):
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


def print_similarity(model):
    origin = ["cat", "barge"]
    identical = [["tiger", "cougar"], ["boat", "vessel"]]
    similar = [["animal", "bird"], ["vehicle", "transport"]]
    different = [["encode", "patriotism"], ["code", "html"]]

    for o, i, s, d in zip(origin, identical, similar, different):
        for i_val in i:
            print(f"{o}|{i_val}: {model.wv.similarity(o, i_val)}")

        for s_val in s:
            print(f"{o}|{s_val}: {model.wv.similarity(o, s_val)}")

        for d_val in d:
            print(f"{o}|{d_val}: {model.wv.similarity(o, d_val)}")


def precompute_idf(term_doc_matrix: sparse.csr_matrix):
    rows, cols = term_doc_matrix.get_shape()
    counts = np.zeros(cols, dtype=np.uint)

    _, cols, _ = sparse.find(term_doc_matrix)
    for col in cols:
        counts[col] += 1

    result = np.array(counts, dtype=float)
    for i, count in enumerate(counts):
        result[i] = math.log10(rows / count)

    return result


def pca_reduce(vectors, vector_size: int):
    pca = PCA(n_components=vector_size)
    return pca.fit_transform(vectors)


def vectorize_test_set(model, _term_doc_idf):
    with open("../../assets/annotated-corpus/test-embeddings.tsv", 'w', newline='') as out_file, \
         open("../../assets/test.csv") as in_file:

        writer = csv.writer(out_file, delimiter='\t')
        reader = csv.reader(in_file)

        sentence_regex = re.compile(sentence_pattern)
        tokenizer = Tokenizer()

        for i, [_, _, text] in enumerate(reader):
            sentences = re.split(sentence_regex, text)

            sentences_vectors = []
            for sentence in sentences:
                tokens = tokenizer.tokenize(sentence)
                tokens_texts = list(map(lambda t: t.text.lower(), tokens))

                tokens_vectors = []
                for token_text in tokens_texts:
                    if token_text not in model.wv:
                        continue

                    token_vector = model.wv[token_text]
                    tokens_vectors.append(token_vector)

                mean_vector = np.mean(tokens_vectors, axis=0)
                sentences_vectors.append(mean_vector)

            doc_vector = np.mean(sentences_vectors, axis=0)
            row = [element for sublist in [[f"doc_id_{i}"], doc_vector] for element in sublist]
            writer.writerow(row)


def cosine_distance(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def main(path: str):
    freq_dict, term_doc_matrix = init_load("../../assets/annotated-corpus/freq-dict.csv",
                                           "../../assets/annotated-corpus/term-doc.npz")
    term_doc_idf = precompute_idf(term_doc_matrix)

    model = gensim.models.Word2Vec.load("../../assets/w2v-data.txt")
    words = ["hello", "cat", "animal", "bird", "code", "html", "vessel", "boat", "encode", "world", "vehicle"]
    tf_idf_vectors = []
    wtv_vectors = []
    for text in words:
        text_term_doc_matrix, text_tf_idf_matrix = get_matrices(text, term_doc_idf, freq_dict)
        text_vector = vectorize_tf_idf_matrix(text_tf_idf_matrix)
        tf_idf_vectors.append(text_vector)
        wtv_vectors.append(model.wv[text])

    reduced = pca_reduce(tf_idf_vectors, model.vector_size)

    cosines = []
    for w_vector, r_vector in zip(wtv_vectors, reduced):
        distance = cosine_distance(w_vector, r_vector)
        cosines.append(distance)
    print(f"{cosines}")

    vectorize_test_set(model, term_doc_idf)


if __name__ == "__main__":
    main(sys.argv[1])
