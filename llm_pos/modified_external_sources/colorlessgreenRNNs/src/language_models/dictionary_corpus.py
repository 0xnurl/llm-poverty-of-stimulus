# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import hashlib
import pathlib
import pickle
from collections import defaultdict

import torch
from loguru import logger


def _get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]


class Dictionary(object):
    def __init__(self, retraining_base_path, dataset_path, experiment_id):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        vocab_path = dataset_path / "vocab.txt"

        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logger.info("Vocab file not found, creating new vocab file...")
            self.create_vocab(
                (
                    dataset_path / "train.txt.original",
                    retraining_base_path / f"train_{experiment_id}.txt",
                )
            )
            logger.info("Done creating new vocab file.")
            open(vocab_path, "w").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        # return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, paths):
        for path in paths:
            with open(path, "r", encoding="utf8") as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        self.add_word(word)


class Corpus(object):
    def __init__(
        self,
        base_path: pathlib.Path,
        dataset_name: str,
        experiment_id: str,
    ):
        logger.info(
            f"Initializing corpus for dataset {dataset_name}, experiment {experiment_id}"
        )

        dataset_path = base_path / dataset_name
        base_train_path = dataset_path / "train.txt.original"
        extra_train_path = base_path / f"{experiment_id}__train.txt"
        base_valid_path = dataset_path / f"valid.txt.original"
        extra_valid_path = base_path / f"{experiment_id}__valid.txt"
        base_test_path = dataset_path / "test.txt.original"
        extra_test_path = base_path / f"{experiment_id}__test.txt"

        base_train_hash = _get_file_hash(base_train_path)
        extra_train_hash = _get_file_hash(extra_train_path)
        train_dict_hash_file = (
            base_path / f"train_dict_{base_train_hash}_{extra_train_hash}.cached"
        )

        if pathlib.Path(train_dict_hash_file).exists():
            with open(train_dict_hash_file, "rb") as f:
                self.dictionary = pickle.load(f)
            logger.info(f"Loaded dictionary from {train_dict_hash_file}")
        else:
            self.dictionary = Dictionary(
                retraining_base_path=base_path,
                dataset_path=dataset_path,
                experiment_id=experiment_id,
            )
            with open(train_dict_hash_file, "wb") as f:
                pickle.dump(self.dictionary, f)
                logger.info(f"Saved dictionary to {train_dict_hash_file}")

        train_paths = [base_train_path, extra_train_path]
        valid_paths = [base_valid_path, extra_valid_path]
        test_paths = [base_test_path, extra_test_path]

        self.train = tokenize(self.dictionary, train_paths)
        self.valid = tokenize(self.dictionary, valid_paths)
        self.test = tokenize(self.dictionary, test_paths)
        logger.info("Done tokenizing.")


def _get_dictionary_hash(dictionary):
    dictionary_str = str(tuple(sorted(dictionary.word2idx.keys())))
    return hashlib.sha1(dictionary_str.encode("utf-8")).hexdigest()[:8]


def _tokenize_file(path, dictionary):
    file_hash = _get_file_hash(path)
    grnn_data_path = path.parent.parent  # .../grnn_data/wikipedia/train.txt.original
    cache_path = grnn_data_path / f"tokenized_ids_{file_hash}.pt"
    if cache_path.exists():
        return torch.load(cache_path)

    ntokens = 0
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            words = line.split()
            ntokens += len(words)

    ids = torch.LongTensor(ntokens)
    token = 0
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            words = line.split()
            for word in words:
                if word in dictionary.word2idx:
                    ids[token] = dictionary.word2idx[word]
                else:
                    ids[token] = dictionary.word2idx["<unk>"]
                token += 1

    torch.save(ids, cache_path)
    return ids


def tokenize(dictionary, paths):
    """Tokenizes a text file for training or testing to a sequence of indices format
    We assume that training and test data has <eos> symbols"""

    logger.info(f"Tokenizing {', '.join(map(str,paths))}...")

    for path in paths:
        assert path.exists(), path

    all_tokenized_ids = []
    for path in paths:
        all_tokenized_ids.append(_tokenize_file(path, dictionary))

    ids = torch.concat(all_tokenized_ids)
    return ids
