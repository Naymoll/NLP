import os
from pathlib import Path

import click

from .tokenizer import format_tokens


@click.group()
def main():
    pass


@main.command()
@click.argument("text", type=str)
def token(text):
    for tk in format_tokens(text):
        print(tk.info())


def replace_path(path, frm, to):
    pre, match, post = path.rpartition(frm)
    return ''.join((to if match else pre, match, post))


@main.command()
@click.argument("path", type=str)
def create_annotations(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            src_file = os.path.join(root, file)
            dist_path_dir = "assets/annotated-corpus/" + '/'.join(src_file.split("/")[2:-1])
            dist_file = os.path.split(src_file)[1] + ".tsv"
            print(dist_path_dir + "/" + dist_file)
            with open(src_file, "r", encoding="utf-8", errors="ignore") as f:
                tokens = format_tokens(f.read())
            if not os.path.exists(dist_path_dir):
                os.makedirs(dist_path_dir)
            with open(dist_path_dir + "/" + dist_file, "w") as f:
                for tk in tokens:
                    f.write(tk.__str__() + "\n")


if __name__ == "__main__":
    main()
