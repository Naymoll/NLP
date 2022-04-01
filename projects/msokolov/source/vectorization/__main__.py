import csv

import nltk
import numpy as np
import gensim

from source.tokenizer.patterns import *
from source.tokenizer import Tokenizer, Token
from source.vectorization import TDMatrix, TextVectorizer
from source.vectorization.util import read_directory

from sklearn.decomposition import PCA


def print_similarity(model):
    origin = ["cat", "hoy"]
    identical = [["tiger", "cougar"], ["boat", "vessel"]]
    similar = [["dog", "bird"], ["auto", "transport"]]
    different = [["rumor", "vendor"], ["code", "html"]]
    for o, i, s, d in zip(origin, identical, similar, different):
        for i_val in i:
            print(f"{o}|{i_val}: {model.wv.similarity(o, i_val)}")

        for s_val in s:
            print(f"{o}|{s_val}: {model.wv.similarity(o, s_val)}")

        for d_val in d:
            print(f"{o}|{d_val}: {model.wv.similarity(o, d_val)}")


def pca_reduce(vectors, vector_size: int):
    pca = PCA(n_components=vector_size)
    return pca.fit_transform(vectors)


def collect_lemmas(tokens: [Token], lemmatizer: nltk.WordNetLemmatizer):
    tokens_lemmas = []
    for token in tokens:
        token.lemmatize(lemmatizer)
        tokens_lemmas.append(token.lemma.lower())

    return tokens_lemmas


def vectorize_sentence(sentence: [Token], model, lemmatizer):
    tokens_lemmas = collect_lemmas(sentence, lemmatizer)
    tokens_vectors = []
    for lemma in tokens_lemmas:
        if lemma not in model.wv:
            continue

        token_vector = model.wv[lemma]
        tokens_vectors.append(token_vector)
    return tokens_vectors


def vectorize_sentences(sentences, model, lemmatizer):
    sentences_vectors = []
    for sentence in sentences:
        tokens_vectors = vectorize_sentence(sentence, model, lemmatizer)

        if len(tokens_vectors) == 0:
            mean_vector = np.zeros(model.vector_size)
        else:
            mean_vector = np.mean(tokens_vectors, axis=0)

        sentences_vectors.append(mean_vector)

    return np.mean(sentences_vectors, axis=0)


def vectorize_test_set(model):
    with open("../../assets/annotated-corpus/test-embeddings.tsv", 'w', newline='') as out_file, \
            open("../../assets/test.csv") as in_file:
        writer = csv.writer(out_file, delimiter='\t')
        reader = csv.reader(in_file)

        tokenizer = Tokenizer([
            words_pattern,
            abbrev_pattern
        ])

        lemmatizer = nltk.WordNetLemmatizer()
        for i, [_, title, text] in enumerate(reader):
            title_sentences = tokenizer.tokenize_sentences(title)
            title_vector = vectorize_sentences(title_sentences, model, lemmatizer)

            text_sentences = tokenizer.tokenize_sentences(text)
            text_vector = vectorize_sentences(text_sentences, model, lemmatizer)

            doc_vector = np.mean([title_vector, text_vector], axis=0)
            row = [element for sublist in [[f"doc_id_{i}"], doc_vector] for element in sublist]
            writer.writerow(row)


def cosine_distance(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def read_corpus(dir_path: str):
    paths = read_directory(dir_path)

    corpus = []
    for path in paths:
        with open(path) as file:
            words = []
            reader = csv.reader(file, delimiter='\t')
            for record in reader:
                if not record:
                    continue

                _, _, stemma = record
                word = stemma.lower()
                words.append(word)
            corpus.append(words)

    return corpus


def main():
    td_matrix = TDMatrix.load("../../assets/annotated-corpus/td-train-data")
    vectorizer = TextVectorizer(td_matrix)
    model = gensim.models.Word2Vec.load("../../assets/w2v-data.txt")

    words = ["hello", "cat", "cargo", "bird", "code", "html", "vessel", "boat", "oil", "world", "year"]
    text_vectors = []
    wtv_vectors = []
    for text in words:
        text_vector = vectorizer.vectorize(text)
        text_vectors.append(text_vector)

        wtv_vector = model.wv[text]
        wtv_vectors.append(wtv_vector)

    reduced = pca_reduce(text_vectors, model.vector_size)
    cosines = []
    for w_vector, r_vector in zip(reduced, wtv_vectors):
        distance = cosine_distance(w_vector, r_vector)
        cosines.append(distance)

    print_similarity(model)
    vectorize_test_set(model)
    print("Done!")


if __name__ == "__main__":
    main()
