import argparse
import pickle

import tqdm
from loguru import logger

import lib

_GPT_J = "gpt-j"


def import_gpt_j_pickle(filename):
    filename = f"gpt_j_{filename}.pickle"
    with open(filename, "rb") as f:
        result_sentences = pickle.load(f)

        for sentence_surprisal in tqdm.tqdm(result_sentences):
            lib.store_in_cache(model=_GPT_J, sentence_surprisal=sentence_surprisal)

    logger.info(f"Imported {len(result_sentences)} to cache.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--filename",
        dest="filename",
        type=str,
    )

    arguments = arg_parser.parse_args()
    import_gpt_j_pickle(
        filename=arguments.filename,
    )
