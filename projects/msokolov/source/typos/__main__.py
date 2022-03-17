import sys
import csv

import source.typos.hirschberg as rust_hirshberg
from source.tokenizer import Tokenizer, Token


DATA_LIMIT = 5000


def load_dictionary(path: str):
    dictionary = set()
    with open(f"{path}") as file:
        reader = csv.reader(file)
        for [word] in reader:
            if word not in dictionary:
                dictionary.add(word)

    return dictionary


def tokenize_record(tokenizer: Tokenizer, record):
    tokens = []
    [label, title, text] = record
    label_tokens = tokenizer.tokenize(label)
    tokens.extend(label_tokens)

    title_tokens = tokenizer.tokenize(title)
    tokens.extend(title_tokens)

    text_tokens = tokenizer.tokenize(text)
    tokens.extend(text_tokens)

    return tokens


def tokenize_from_files(ok_path: str, err_path):
    tokenizer = Tokenizer()
    ok_tokens = []
    err_tokens = []
    with open(ok_path) as ok_file, open(err_path) as err_file:
        ok_reader = csv.reader(ok_file)
        err_reader = csv.reader(err_file)
        for ok_record, err_record in zip(ok_reader, err_reader):
            ok_record_tokens = tokenize_record(tokenizer, ok_record)
            err_record_tokens = tokenize_record(tokenizer, err_record)
            if len(ok_record_tokens) != len(err_record_tokens):
                continue

            ok_tokens.extend(ok_record_tokens)
            err_tokens.extend(err_record_tokens)
            if len(ok_tokens) >= DATA_LIMIT:
                break

    return ok_tokens, err_tokens


def check_tokens(correct: [Token], incorrect: [Token]):
    count = 0
    for cor_token, incor_token in zip(correct, incorrect):
        if cor_token == incor_token:
            count += 1

    return count


def print_stats(tokens: [Token], before: int, after: int):
    tokens_len = len(tokens)
    print(f"Total: {tokens_len}\n"
          f"Correct tokens before: {before}/{(before / tokens_len):.3f}\n"
          f"Correct tokens after: {after}/{(after / tokens_len):.3f}\n"
          f"Diff: {after - before}")


def write_result(tokens: [Token]):
    with open(f"../../assets/correct.csv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(map(lambda t: t.text, tokens))


def main(ok_path: str, err_path: str):
    dictionary = load_dictionary("../../assets/dictionary.csv")

    tokens_from_ok, tokens_from_err = tokenize_from_files(ok_path, err_path)
    cor_tokens_before = check_tokens(tokens_from_ok, tokens_from_err)

    dictionary_list = list(dictionary)
    for token in tokens_from_err:
        token_text = token.text
        if token_text in dictionary:
            continue

        token.text = rust_hirshberg.try_correct(dictionary_list, token_text)

    cor_tokens_after = check_tokens(tokens_from_ok, tokens_from_err)
    print_stats(tokens_from_err, cor_tokens_before, cor_tokens_after)
    write_result(tokens_from_err)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
