import argparse
import math
import os
import random
import subprocess

from loguru import logger

import lib

_BASE_SEED = 1111

_TRANSFORMER_DEFAULT_CONTEXT_SIZE = 30


def _experiment_id(
    phenomenon: str,
    train_ratio: float,
    grammar_idx: int,
    seed: int,
    grammar_version: str,
):
    return f"{phenomenon}__ratio_{str(train_ratio).replace('.', '_')}__grammar_{grammar_idx if grammar_idx is not None else 'all'}__seed_{seed}_v{grammar_version}"


def _assert_no_shared_sentences(phenomenon, names_size):
    retraining_sentences = frozenset(
        lib.iterate_phenomenon_sentences(
            phenomenon,
            names_size,
            conditions=(lib.PLUS_FILLER_PLUS_GAP,),
            retraining_corpus=True,
        )
    )
    main_sentences = frozenset(
        lib.iterate_phenomenon_sentences(
            phenomenon,
            names_size,
            conditions=(lib.PLUS_FILLER_PLUS_GAP,),
            retraining_corpus=False,
        )
    )
    assert len(retraining_sentences & main_sentences) == 0


def _gen_data_from_retraining_grammar(
    phenomenon,
    names_size,
    seed,
    samples_per_condition,
    experiment_id,
):
    random.seed(seed)

    retraining_sentences = []
    for condition in (
        lib.PLUS_FILLER_PLUS_GAP,
        lib.MINUS_FILLER_MINUS_GAP,
    ):
        condition_sentences = tuple(
            lib.iterate_phenomenon_sentences(
                phenomenon,
                names_size,
                conditions=(condition,),
                retraining_corpus=True,
            )
        )
        condition_sample = random.sample(condition_sentences, k=samples_per_condition)
        retraining_sentences += condition_sample

    _assert_no_shared_sentences(phenomenon, names_size)

    num_total_sentences = len(retraining_sentences)

    # Using test-size = 0 because there's no need for a test set in our setup. The test set is the main CFG.
    test_size = 0

    # 8-to-1 train/validation proportion based on Gulordava et al.
    train_size = math.ceil(len(retraining_sentences) * 8 / 9)

    train_idxs = random.sample(range(num_total_sentences), k=train_size)
    val_idxs = [i for i in range(num_total_sentences) if i not in set(train_idxs)]

    train_sentences = [retraining_sentences[i] for i in train_idxs]
    val_sentences = [retraining_sentences[i] for i in val_idxs]
    test_sentences = []

    assert len(train_sentences) == train_size
    assert len(test_sentences) == test_size
    assert len(set(train_sentences) & set(val_sentences)) == 0
    assert len((set(train_sentences) & set(val_sentences)) & (set(test_sentences))) == 0

    logger.info(
        f"Training sentences: {len(train_sentences)}, val: {len(val_sentences)}, test: {len(test_sentences)}."
    )

    _save_retraining_data(
        train_sentences=train_sentences,
        val_sentences=val_sentences,
        test_sentences=test_sentences,
        experiment_id=experiment_id,
    )


def _save_retraining_data(
    train_sentences, val_sentences, test_sentences, experiment_id
):
    training_path = lib.RETRAINING_DATA_PATH / f"{experiment_id}__train.txt"
    val_path = lib.RETRAINING_DATA_PATH / f"{experiment_id}__valid.txt"
    test_path = lib.RETRAINING_DATA_PATH / f"{experiment_id}__test.txt"

    with training_path.open("w") as f:
        for sentence in train_sentences:
            f.write(f"{sentence} <eos>\n")

    with val_path.open("w") as f:
        for sentence in val_sentences:
            f.write(f"{sentence} <eos>\n")

    with test_path.open("w") as f:
        for sentence in test_sentences:
            f.write(f"{sentence} <eos>\n")

    vocab_path = lib.RETRAINING_DATA_PATH / "vocab.txt"
    if vocab_path.exists():
        logger.warning(f"Using existing vocab.txt")


def _train_transformer(
    experiment_id: str,
    dataset_name: str,
    context_size,
    seed: int,
):
    logger.info(f"Retraining Transformer with arguments: {vars(arguments)}")

    command_args = (
        [
            "python",
            "./modified_external_sources/lm_povstim_with_childes/main.py",
            "--nlayers",
            "8",
            "--bptt",
            str(context_size),
            "--nhid",
            "1600",
            "--emsize",
            "800",
            "--lr",
            "5.0",
            "--batch_size",
            "10",
            "--dropout",
            "0.2",
            "--nhead",
            "16",
            "--model",
            "Transformer",
            "--data",
            str(lib.RETRAINING_DATA_PATH),
            "--dataset_name",
            dataset_name,
            "--seed",
            str(seed),
            "--log",
            f"log_transformer_{experiment_id}.txt",
            "--experiment-id",
            experiment_id,
        ]
        + (["--cuda"] if os.getenv("CUDA") else [])
        + (["--gpu", arguments.gpu_id] if arguments.gpu_id is not None else [])
    )

    logger.info("Running command:")
    logger.info(str(command_args))

    subprocess.run(command_args)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p",
        "--phenomenon",
        dest="phenomenon",
        help=f"Phenomenon name: 'PG', 'ATB', 'SUBJECT_AUX'",
    )
    arg_parser.add_argument(
        "-t",
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help=f"Train ratio, e.g. 0.3.",
    )
    arg_parser.add_argument(
        "--names",
        dest="names_size",
        help=f"Names size: 'tiny', 'small', 'large'",
    )
    arg_parser.add_argument(
        "--grammar_idx",
        type=int,
        default=None,
        help=f"Grammar to generate phenomenon sentences from. Default: None (generate from all available grammars)",
    )

    arg_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=_BASE_SEED,
        help=f"Seed. Default: {_BASE_SEED}",
    )
    arg_parser.add_argument(
        "--context_size",
        type=int,
        default=_TRANSFORMER_DEFAULT_CONTEXT_SIZE,
    )
    arg_parser.add_argument(
        "--samples_per_condition",
        type=int,
        default=None,
    )
    arg_parser.add_argument(
        "--dataset_name",
        type=str,
        help="'wikipedia' or 'childes'",
    )
    arg_parser.add_argument(
        "--gpu",
        dest="gpu_id",
        default=None,
    )
    arguments = arg_parser.parse_args()

    grammar_version = lib.get_grammar_version()
    experiment_id = f"{arguments.phenomenon}__retraining__dataset_{arguments.dataset_name}__seed_{arguments.seed}__v{grammar_version}__{arguments.names_size}__samples_{arguments.samples_per_condition}"

    _gen_data_from_retraining_grammar(
        phenomenon=arguments.phenomenon,
        names_size=arguments.names_size,
        seed=arguments.seed,
        samples_per_condition=arguments.samples_per_condition,
        experiment_id=experiment_id,
    )

    _train_transformer(
        experiment_id=experiment_id,
        dataset_name=arguments.dataset_name,
        context_size=arguments.context_size,
        seed=arguments.seed,
    )
