import argparse
import os
import pathlib
import pickle
from datetime import datetime

import torch
import transformers
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

import lib
import run_gen_all_sentences

_BATCH_SIZE = 100
_SEED = 100
_TOKENIZER_NEW_WORD_PREFIX = "Ġ"


_DEVICE = "cuda"
model_path = str(
    pathlib.Path(os.environ["DSDIR"]) / "HuggingFace_Models/EleutherAI/gpt-j-6B/"
)
dtype = torch.float16
revision = "float16"
model_class = AutoModelForCausalLM
model_name = "gpt-j"


def _run_inference_on_sentence_file(filename: str):
    logger.info(f"Running on file {filename}")
    logger.info("Loading model...")

    model = model_class.from_pretrained(
        model_path, revision=revision, torch_dtype=dtype
    ).to(_DEVICE)

    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    with open(f"{filename}.txt", "r") as f:
        sentences = list(map(str.strip, f.readlines()))

    results = []

    for i in range(0, len(sentences), _BATCH_SIZE):
        logger.info(f"Batch {i}-{i+_BATCH_SIZE} / {len(sentences)}")
        sentence_batch = sentences[i : i + _BATCH_SIZE]

        max_sentence_length = len(
            max(list(map(lambda x: x.split(" "), sentence_batch)), key=len)
        )

        logger.info(f"Max sentence length: {max_sentence_length}")

        logger.info("Tokenizing...")
        input_ids = tokenizer(
            sentence_batch, return_tensors="pt", padding=True
        ).input_ids.to(_DEVICE)

        if input_ids.shape[1] > max_sentence_length + 1:
            logger.warning("Tokenized shape > max sentence length + 1")

        logger.info("Feeding model...")

        start_time = datetime.now()

        model.eval()
        with torch.no_grad():
            logits = model(input_ids).logits

        logits = logits.cpu()

        log_softmax = torch.log_softmax(logits, dim=-1)

        input_ids = input_ids.cpu()

        next_word_probabs = torch.take_along_dim(
            log_softmax[:, :-1], input_ids[:, 1:].unsqueeze(-1), dim=-1
        )

        surprisals = -torch.concat(
            (
                torch.zeros((next_word_probabs.shape[0], 1, 1)),
                next_word_probabs,
            ),
            dim=1,
        )

        end_time = datetime.now()
        delta_seconds = (end_time - start_time).total_seconds()

        logger.info(
            f"Inference took {delta_seconds} seconds = {delta_seconds / len(sentence_batch)} seconds/sentence."
        )

        assert logits.shape[0] == len(sentence_batch)

        for b in range(logits.shape[0]):
            sentence_ids = input_ids[b]

            tokenizer_tokens = tokenizer.convert_ids_to_tokens(
                sentence_ids,
                skip_special_tokens=True,
            )
            current_surprisal_tokens = []

            current_token_idx = -1
            for i, token in enumerate(tokenizer_tokens):
                if current_token_idx == -1 or token.startswith(
                    _TOKENIZER_NEW_WORD_PREFIX
                ):
                    current_token_idx += 1

                token_str_clean = token.replace(_TOKENIZER_NEW_WORD_PREFIX, "")
                current_surprisal_tokens.append(
                    lib.Token(
                        text=token_str_clean,
                        surprisal=surprisals[b, i].item(),
                        idx=current_token_idx,
                    )
                )

            tokenizer_decoded_tokens = tuple(
                tokenizer.decode(
                    sentence_ids,
                    skip_special_tokens=True,
                ).split(" ")
            )
            original_sentence = lib.tokens_to_sentence(tokenizer_decoded_tokens)

            sentence_surprisal = lib.SentenceSurprisal(
                tokens=tuple(current_surprisal_tokens),
                model=model_name,
                original_sentence=original_sentence,
            )
            results.append(sentence_surprisal)

        with open(f"gpt_j_{filename}.pickle", "wb") as f:
            pickle.dump(results, f)

        del sentence_ids
        del surprisals
        torch.cuda.empty_cache()


def _run_phenomenon(phenomenon, names_size):
    transformers.set_seed(_SEED)

    run_gen_all_sentences.write_sentences_plaintext(
        phenomenon=phenomenon, names_size=names_size
    )
    grammar_version = lib.get_grammar_version()
    filename = f"all_sentences_{phenomenon}_{names_size}_{grammar_version}"

    _run_inference_on_sentence_file(filename)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p",
        "--phenomenon",
        dest="phenomenon",
        type=str,
    )
    arg_parser.add_argument(
        "-s",
        "--size",
        dest="names_size",
        type=str,
    )
    arg_parser.add_argument(
        "--filename",
        dest="filename",
        type=str,
        default=None,
    )
    arguments = arg_parser.parse_args()

    if arguments.filename is not None:
        _run_inference_on_sentence_file(arguments.filename)
    else:
        _run_phenomenon(
            phenomenon=arguments.phenomenon, names_size=arguments.names_size
        )
