import os.path
import sys
import numpy as np
import csv
import matplotlib.colors

import matplotlib.pyplot as plt
import sklearn.metrics


from source.tokenizer import Tokenizer
from source.vectorization.__main__ import create_term_doc_matrix

from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics import r2_score


OUT_PATH = "../../assets/topic-modeling"


def read_freq_dict(path: str):
    freq_dict = {}
    with open(path) as file:
        reader = csv.reader(file)
        for [token, freq] in reader:
            freq_dict[token] = freq

    return freq_dict


def get_top_topics_words(model: LDA, words: [str]):
    print(f"Start get_top_topics_words")
    topics_words = []
    for topic in model.components_:
        top_features_ind = topic.argsort()[: -10 - 1: -1]
        topic_words = [words[i] for i in top_features_ind]
        topics_words.append(topic_words)

    return topics_words


def write_top_words(topics_words, iter: int, topic: int):
    print(f"Start write_top_words")
    with open(f"{OUT_PATH}/top-ten-{iter}-{topic}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(topics_words)


def write_doc_topic_distr(doc_topic_distr: np.array, iter: int, topic: int):
    print("Start write_doc_topic_distr")
    with open(f"{OUT_PATH}/doc-topic-distr-{iter}-{topic}.tsv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(doc_topic_distr)


def read_test_td_matrix(freq_dict: dict[str, int]):
    corpus = []
    with open("../../assets/test.csv") as file:
        reader = csv.reader(file)

        tokenizer = Tokenizer()
        for i, [_, _, text] in enumerate(reader):
            tokens = tokenizer.tokenize(text)
            tokens_texts = list(map(lambda t: t.text.lower(), tokens))
            corpus.append(tokens_texts)

    return create_term_doc_matrix(corpus, freq_dict)


def choose_y(x, y, deg):
    return np.poly1d(np.polyfit(x, y, deg))(x)


def choose_poly_deg(x, y):
    prev = -0.1
    cur = 0.0
    deg = 0
    while cur > prev:
        deg += 1
        prev = cur
        cur = r2_score(y, choose_y(x, y, deg))

    return deg - 1


def create_plot():
    with open(f"{OUT_PATH}/perplexities.csv") as file:
        reader = csv.reader(file)

        data = {}
        for [i, topic, perplexity] in reader:
            if i not in data:
                data[i] = ([], [])

            data[i][0].append(int(topic))
            data[i][1].append(float(perplexity))

    colors = iter(matplotlib.colors.BASE_COLORS)
    for (i, coords) in data.items():
        color = next(colors)
        x, y = coords[0], coords[1]
        plt.plot(x, y, '-o', linewidth="0.3", color=color, label=f"Iter: {i}")

        deg = choose_poly_deg(x, y)
        plt.plot(x, choose_y(x, y, deg), linewidth="1", color=next(colors), label=f"Iter poly: {i}")

    plt.legend(loc="upper left")
    plt.xlabel('Topics')
    plt.ylabel('Perplexities')
    plt.show()


def main(path: str):
    train_td_matrix = sparse.load_npz(path)
    freq_dict = read_freq_dict("../../assets/annotated-corpus/freq-dict.csv")

    test_td_matrix_path = "../../assets/test-td-matrix.npz"
    if os.path.isfile(test_td_matrix_path):
        test_td_matrix = sparse.load_npz(test_td_matrix_path)
    else:
        test_td_matrix = read_test_td_matrix(freq_dict)
        sparse.save_npz(test_td_matrix_path, test_td_matrix)

    words = list(map(lambda entry: entry[0], freq_dict.items()))

    topics = [2, 4, 8, 16, 32]
    iters = [2, 4, 8]

    with open(f"{OUT_PATH}/perplexities.csv", 'w', newline='') as file:
        writer = csv.writer(file)

        for iter in iters:
            for topic in topics:
                print(f"Start {iter} iter/{topic} topic")

                lda = LDA(n_components=topic, max_iter=iter, n_jobs=-1)
                doc_topic_distr = lda.fit_transform(train_td_matrix)
                write_doc_topic_distr(doc_topic_distr, iter, topic)

                top_topics_words = get_top_topics_words(lda, words)
                write_top_words(top_topics_words, iter, topic)

                print(f"Start calculate perplexity")
                perplexity = lda.perplexity(test_td_matrix)
                writer.writerow([iter, topic, perplexity])

    create_plot()


if __name__ == "__main__":
    main(sys.argv[1])
    # create_plot()