import os
import pathlib

import torch
from loguru import logger


class Dictionary(object):
    # Init with path to vocab file
    def __init__(self, path=None):
        self.word2idx = {}
        self.idx2word = []

        # vocab.txt should list the vocab, one
        # word per line
        vocab_path = pathlib.Path("../models/yedetore/vocab.txt")
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logger.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, "train.txt"))
            open(vocab_path, "w").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    self.add_word(word)


def split_possesives_and_contractions(word):
    if word.endswith("'s"):
        return word[:-2] + " 's"
    if word == "can't":
        return "can n't"
    if word.endswith("n't"):
        return word[:-3] + " n't"
    if word.endswith("'re"):
        return word[:-3] + " 're"
    if word.endswith("'m"):
        return word[:-2] + " 'm"
    if word.endswith("'d"):
        return word[:-2] + " 'd"
    if word.endswith("'ll"):
        return word[:-3] + " 'll"
    if word.endswith("'ve"):
        return word[:-3] + " 've"
    if word.endswith("s'"):
        return word[:-1] + " '"
    if word.endswith("'r"):
        return word[:-2] + " are"
    if word.endswith("'has"):
        return word[:-4] + " has"
    if word.endswith("'is"):
        return word[:-3] + " is"
    if word.endswith("'did"):
        return word[:-4] + " did"
    if word == "wanna":
        return "want to"
    if word == "hafta":
        return "have to"
    if word == "gonna":
        return "going to"
    if word == "okay":
        return "ok"
    if word == "y'all":
        return "you all"
    if word == "c'mere":
        return "come here"
    if word == "I'ma":
        return "I am going to"
    if word == "what'cha":
        return "what are you"
    if word == "don'tcha":
        return "do you not"

    # List of startswith exceptions: ["t'", "o'", "O'", "d'"]
    # List of == exceptions: ["Ma'am", "ma'am", "An'", "b'ring", "Hawai'i","don'ting", "rock'n'roll" "don'ting", "That'scop","that'ss","go'ed", "s'pose", "'hey", "me'", "shh'ell", "th'do", "Ross'a", "him'sed"]
    # List of in exceptions: ["_", "-"]
    # List of endswith exceptions (note that this one is a catch all condition): ["'"]

    return word
