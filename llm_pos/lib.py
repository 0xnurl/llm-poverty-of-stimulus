import collections
import csv
import dataclasses
import hashlib
import io
import os
import pathlib
import pickle
import re
import subprocess
import time
from typing import Iterable, Optional

import cachetools
import dataclasses_json
import nltk
import numpy as np
import pandas as pd
import toml
import torch
import transformers
from loguru import logger
from nltk.parse import earleychart, generate
from tqdm import tqdm

import utils
from modified_external_sources.yedetore import vocab as childes_vocab

DATA_PATH = pathlib.Path("../data/")
RETRAINING_DATA_PATH = DATA_PATH / "retraining"

_GRNN_MOUNT_INSIDE_DOCKER = pathlib.Path("/tmp/grnn-mount/")

_DOCKER_BASE_IMAGE = "cpllab/language-models"

_UNGRAMMATICAL = "*"

_SENTENCE_ID = "sentence_id"
_TOKEN_ID = "token_id"
_TOKEN = "token"
_SURPRISAL = "surprisal"

_EOS = "EOS"
_SOS = "SOS"
_SENTINEL = "sentinel"

PADDING = "<pad>"
CRITICAL_REGION_MARKER = "_"
CRITICAL_REGION_DERIVATION_SYMBOL = "X"

_JRNN = "jrnn"
_GRNN = "grnn"
_GPT2 = "gpt2"
GPT3 = "gpt3"

# https://platform.openai.com/docs/models.
# https://openai.com/pricing#language-models.
_GPT3_CHEAP_ENGINE = "text-ada-001"
GPT3_DEFAULT_ENGINE = "text-davinci-003"

_MODEL_SYMBOLS = {
    _JRNN: {
        _SOS: "<S>",
        _EOS: "</S>",
        _SENTINEL: None,
    },
    _GRNN: {
        _SOS: None,
        _EOS: "<eos>",
        _SENTINEL: None,
    },
    _GPT2: {
        _SOS: "<|endoftext|>",
        _EOS: "<|endoftext|>",
        _SENTINEL: "Ġ",
    },
}


_FLOAT = np.float64

PLUS_FILLER_PLUS_GAP = "PLUS_FILLER_PLUS_GAP"
MINUS_FILLER_PLUS_GAP = "MINUS_FILLER_PLUS_GAP"
PLUS_FILLER_MINUS_GAP = "PLUS_FILLER_MINUS_GAP"
MINUS_FILLER_MINUS_GAP = "MINUS_FILLER_MINUS_GAP"

CONDITIONS = (
    PLUS_FILLER_PLUS_GAP,
    MINUS_FILLER_PLUS_GAP,
    PLUS_FILLER_MINUS_GAP,
    MINUS_FILLER_MINUS_GAP,
)

FRIENDLY_CONDITION_NAME = {
    PLUS_FILLER_PLUS_GAP: "+G,+F",
    PLUS_FILLER_MINUS_GAP: "-G,+F",
    MINUS_FILLER_PLUS_GAP: "+G,-F",
    MINUS_FILLER_MINUS_GAP: "-G,-F",
}


def _get_redis_cache():
    try:
        import redis
    except ModuleNotFoundError:
        return None

    r = redis.Redis(host="localhost", port=6379, db=0)
    try:
        r.get("")
    except redis.exceptions.ConnectionError:
        logger.warning("Redis server not running")
        return None
    return r


_SURPRISAL_CACHE = _get_redis_cache() or cachetools.LRUCache(maxsize=1_000_000)

_TORCH_MODELS_CACHE = cachetools.LRUCache(maxsize=3)
_CORPUS_CACHE = cachetools.LRUCache(maxsize=3)


@dataclasses_json.dataclass_json()
@dataclasses.dataclass(frozen=True)
class Token:
    text: str
    idx: int
    surprisal: Optional[_FLOAT] = None
    critical_region: bool = False


@dataclasses_json.dataclass_json()
@dataclasses.dataclass(frozen=True)
class Sentence:
    original_token_strings: tuple[str, ...]
    preprocessed_tokens: tuple[Token, ...]
    padding_idxs: frozenset[int]
    grammatical: bool

    def __str__(self):
        return _join(x.text for x in self.preprocessed_tokens)


@dataclasses_json.dataclass_json()
@dataclasses.dataclass(frozen=True)
class SentenceSurprisal:
    tokens: tuple[Token, ...]
    model: str
    original_sentence: Sentence

    def __sub__(self, other: "SentenceSurprisal"):
        return _subtract_surprisals(self, other)

    def __str__(self):
        return str(self.original_sentence)


def get_grammar_version():
    return load_grammars_toml()["VERSION"]


def _get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]


def get_surprisal_dataframe_id(
    phenomenon, model, names_size, grammar_version, sentence_file, grammar_idx
):
    s = f"{phenomenon}__model__{model}__cfg_{names_size}_v{grammar_version}"

    if grammar_idx is not None:
        s += f"__grammar_{grammar_idx}"

    if sentence_file:
        s += f"__intersected_with_{pathlib.Path(sentence_file).stem}"

    return s


def load_surprisal_dataframe(surprisal_df_id) -> pd.DataFrame:
    with open(DATA_PATH / f"{surprisal_df_id}.pickle", "rb") as f:
        return pickle.load(f)


def get_critical_words(sentence: Sentence) -> tuple[str, ...]:
    return tuple(x.text for x in sentence.preprocessed_tokens if x.critical_region)


def get_critical_surprisals(surprisal: SentenceSurprisal) -> tuple[_FLOAT, ...]:
    return tuple(x.surprisal for x in surprisal.tokens if x.critical_region)


def _pad_sentence(s: SentenceSurprisal, pad_size: int) -> SentenceSurprisal:
    pad_start_idx = s.tokens[-1].idx + 1
    pad_tokens = tuple(
        Token(text=PADDING, surprisal=np.nan, idx=i)
        for i in range(pad_start_idx, pad_start_idx + pad_size)
    )
    return dataclasses.replace(s, tokens=s.tokens + pad_tokens)


def pad(surprisals: tuple[SentenceSurprisal, ...]) -> tuple[SentenceSurprisal, ...]:
    token_lengths = tuple(map(lambda x: len(x.tokens), surprisals))
    shortest = min(token_lengths)
    longest = max(token_lengths)

    if longest - shortest == 0:
        return surprisals

    padded = []
    for s in surprisals:
        if len(s.tokens) == longest:
            padded_s = s
        else:
            padded_s = _pad_sentence(s, pad_size=longest - len(s.tokens))
        padded.append(padded_s)
    return tuple(padded)


@dataclasses_json.dataclass_json()
@dataclasses.dataclass(frozen=True)
class SurprisalDiff:
    sentence1: SentenceSurprisal
    sentence2: SentenceSurprisal
    surprisal_diffs: tuple[_FLOAT, ...]


def _subtract_surprisals(s1, s2) -> SurprisalDiff:
    if len(s1.tokens) != len(s2.tokens):
        pad_size = abs(len(s1.tokens) - len(s2.tokens))
        if len(s1.tokens) < len(s2.tokens):
            s1 = _pad_sentence(s1, pad_size)
        else:
            s2 = _pad_sentence(s2, pad_size)

    surprisal_diffs = tuple(
        map(
            lambda a: a[0] - a[1],
            zip(
                tuple(x.surprisal for x in s1.tokens),
                tuple(x.surprisal for x in s2.tokens),
            ),
        )
    )

    return SurprisalDiff(sentence1=s1, sentence2=s2, surprisal_diffs=surprisal_diffs)


def format_sentence(
    sentence: Sentence,
    add_grammaticality,
    left_region_marker,
    right_region_marker,
    add_critical_surprisal,
) -> str:
    text = ""
    if add_grammaticality and not sentence.grammatical:
        text += "* "
    for t, token in enumerate(sentence.preprocessed_tokens):
        if token.critical_region and len(left_region_marker):
            text += f"{left_region_marker} "

        text += token.text

        if token.critical_region and len(right_region_marker):
            text += f" {right_region_marker}"

        if token.critical_region and add_critical_surprisal:
            text += f" ({token.surprisal:.2f})"

        if t < len(sentence.preprocessed_tokens) - 1:
            text += " "
    return text


def _cleanup_result_token(token: str, model: str) -> str:
    if _MODEL_SYMBOLS[model][_SENTINEL] is not None:
        token = token.replace(_MODEL_SYMBOLS[model][_SENTINEL], "")
    return token


def _is_new_word(token: str, model: str) -> bool:
    model_sentinel = _MODEL_SYMBOLS[model][_SENTINEL]
    if model_sentinel is None or token.startswith(model_sentinel):
        if model in {_GRNN, _JRNN} and token.startswith("'"):
            # GRNN and JRNN break "Mary's" into "Mary" + "'s" + they don't have a special symbol ("sentinel") to delimit words.
            return False

        return True

    return False


def _csv_output_to_surprisals(
    output: str, model: str, original_sentences: tuple[Sentence, ...]
) -> tuple[SentenceSurprisal, ...]:
    model_base, _ = get_model_base_and_checkpoint(model)

    csv_reader = csv.DictReader(
        f=io.StringIO(output),
        delimiter="\t",
    )
    surprisals = []
    current_tokens = []
    current_token_idx = -1
    current_sentence_idx = 1

    for row in csv_reader:
        sentence_id = int(row[_SENTENCE_ID])

        if sentence_id > 1 and sentence_id != current_sentence_idx:
            surprisals.append(
                SentenceSurprisal(
                    tokens=tuple(current_tokens),
                    model=model,
                    original_sentence=original_sentences[current_sentence_idx - 1],
                )
            )
            current_tokens = []
            current_token_idx = -1
            current_sentence_idx = sentence_id

        token_str = row[_TOKEN]
        token_str_clean = _cleanup_result_token(row[_TOKEN], model_base)

        if token_str_clean in {
            _MODEL_SYMBOLS[model_base][_SOS],
            _MODEL_SYMBOLS[model_base][_EOS],
        }:
            continue

        if current_token_idx == -1 or _is_new_word(token_str, model_base):
            current_token_idx += 1

        current_tokens.append(
            Token(
                text=token_str_clean,
                surprisal=_FLOAT(row[_SURPRISAL]),
                idx=current_token_idx,
            )
        )

    surprisals.append(
        SentenceSurprisal(
            tokens=tuple(current_tokens),
            model=model,
            original_sentence=original_sentences[current_sentence_idx - 1],
        )
    )

    return tuple(surprisals)


def _string_to_sentence(s: str) -> Sentence:
    return tokens_to_sentence(tokenize(s))


def get_surprisals_for_strings(
    strings: tuple[str, ...], model: str
) -> SentenceSurprisal:
    return get_surprisals_per_model(strings, (model,))[model]


def get_surprisals_per_model(
    sentences: tuple[str, ...], models: tuple[str, ...]
) -> dict[str, tuple[SentenceSurprisal, ...]]:
    preprocessed_sentences = tuple(map(_string_to_sentence, sentences))
    surprisals_per_model = {}
    for model in models:
        surprisals = get_surprisals(preprocessed_sentences, model)
        surprisals = tuple(map(_reintroduce_padding_tokens, surprisals))
        surprisals_per_model[model] = surprisals
    return surprisals_per_model


def _kill_container(container_id: str):
    subprocess.run(["docker", "kill", container_id])


def _get_running_container_id(model: str) -> Optional[str]:
    model_base, checkpoint = get_model_base_and_checkpoint(model)

    ps_output = subprocess.check_output(
        [
            "docker",
            "ps",
            "--filter",
            f"ancestor={_DOCKER_BASE_IMAGE}:{model_base}",
            "--filter",
            f"label=checkpoint={checkpoint}",
        ]
    ).decode("utf-8")
    containers = ps_output.strip().split("\n")[1:]
    if len(containers) == 0:
        return None
    container_id = containers[0].split(" ")[0]
    logger.debug(
        f"Container for model {model_base} checkpoint {checkpoint} already running: {container_id}"
    )
    return container_id


def _start_container(model: str) -> str:
    model_base, checkpoint = get_model_base_and_checkpoint(model)
    post_command = None

    if model_base == _JRNN:
        post_command = "cp /tmp/pos-mount/modified_external_sources/lm-zoo/JRNN/bin/get_surprisals_concurrent /opt/bin/get_surprisals"

    elif model_base == _GRNN and checkpoint is not None:
        checkpoint_path = _GRNN_MOUNT_INSIDE_DOCKER / "checkpoints" / f"{checkpoint}.pt"
        post_command = f"cp {checkpoint_path} /opt/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt"

    full_container_id = (
        subprocess.check_output(
            [
                "docker",
                "run",
                "-i",
                "-d",
                "--rm",
                "-v",
                f"{RETRAINING_DATA_PATH}:{_GRNN_MOUNT_INSIDE_DOCKER}",
                "-v",
                f"{os.getcwd()}:/tmp/pos-mount/",
                "--label",
                f"checkpoint={checkpoint}",
                f"{_DOCKER_BASE_IMAGE}:{model_base}",
                f"/bin/bash",
            ],
        )
        .decode("utf-8")
        .split("\n")[0]
    )
    short_id = full_container_id[:12]

    if post_command:
        process = subprocess.run(
            [
                "docker",
                "exec",
                short_id,
                "/bin/bash",
                "-c",
                post_command,
            ],
            capture_output=True,
        )
        if process.returncode != 0:
            _kill_container(short_id)
            raise RuntimeError(
                f"Error running Docker command on {short_id}: {process.stderr.decode('utf-8')}"
            )

    logger.debug(
        f"Started container for model {model_base} checkpoint {checkpoint}: container id {short_id}"
    )
    return short_id


def _start_or_get_container(model: str) -> str:
    container_id = _get_running_container_id(model)
    if container_id is None:
        container_id = _start_container(model)
    return container_id


def tokenize(s) -> tuple[str, ...]:
    s = re.sub(r"(\w)([\?\.\.,])", r"\1 \2", s)
    return tuple(s.split(" "))


def _join(tokens) -> str:
    return " ".join(tokens)


def _extract_padding_tokens(tokens) -> tuple[frozenset[int], tuple[str, ...]]:
    padding_idxs = set()
    non_padding_tokens = []
    for i, t in enumerate(tokens):
        if t == PADDING:
            padding_idxs.add(i - len(padding_idxs))
        else:
            non_padding_tokens.append(t)
    return frozenset(padding_idxs), tuple(non_padding_tokens)


def _extract_critical_region_idxs(tokens) -> tuple[frozenset[int], tuple[str, ...]]:
    critical_region_idxs = set()
    non_marker_tokens = []

    in_critical_region = False

    curr_token_idx = 0

    for token in tokens:
        if token == CRITICAL_REGION_MARKER:
            in_critical_region = not in_critical_region
            continue

        non_marker_tokens.append(token)

        if in_critical_region:
            critical_region_idxs.add(curr_token_idx)

        curr_token_idx += 1

    return frozenset(critical_region_idxs), tuple(non_marker_tokens)


def tokens_to_sentence(token_strings: tuple[str, ...]) -> Sentence:
    grammatical = True
    if token_strings[0] == _UNGRAMMATICAL:
        token_strings = token_strings[1:]
        grammatical = False

    # TODO: Fix bug: if critical region appears before padding wrong idxs are saved.

    paddings_idxs, tokens_without_pads = _extract_padding_tokens(token_strings)

    critical_region_idxs, tokens_without_markers = _extract_critical_region_idxs(
        tokens_without_pads
    )

    preprocessed_tokens = []
    for t, token_str in enumerate(tokens_without_markers):
        preprocessed_tokens.append(
            Token(text=token_str, critical_region=t in critical_region_idxs, idx=t)
        )

    return Sentence(
        original_token_strings=token_strings,
        preprocessed_tokens=tuple(preprocessed_tokens),
        padding_idxs=paddings_idxs,
        grammatical=grammatical,
    )


def _get_surprisals_from_openai(
    sentences: tuple[Sentence, ...], engine: str
) -> tuple[SentenceSurprisal, ...]:
    import openai

    openai.api_key = os.getenv("OPENAI_KEY")

    time.sleep(1)  # Ugly way to throttle requests.

    response = openai.Completion.create(
        engine=engine,
        echo=True,
        prompt=list(str(x) for x in sentences),
        temperature=0.7,
        max_tokens=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1,
    )

    results = []
    choices = response["choices"]
    for c, choice in enumerate(choices):
        index = choice["index"]
        logprobs = choice["logprobs"]
        token_logprobs = logprobs["token_logprobs"]
        tokens = logprobs["tokens"]

        current_token_idx = 0
        surprisal_tokens = []
        for t, token_text in enumerate(tokens):
            token_logprob = token_logprobs[t]
            if token_logprob is not None:
                surprisal = -1 * token_logprob
            else:
                surprisal = 0.0

            if token_text.startswith(" "):
                current_token_idx += 1

            surprisal_tokens.append(
                Token(
                    text=token_text.strip(), surprisal=surprisal, idx=current_token_idx
                )
            )

        results.append(
            SentenceSurprisal(
                tokens=tuple(surprisal_tokens),
                model=f"gpt3__{engine}",
                original_sentence=sentences[c],
            )
        )

    return tuple(results)


def _reintroduce_padding_tokens(
    surprisal: SentenceSurprisal,
) -> SentenceSurprisal:
    tokens = surprisal.tokens
    new_tokens = list(tokens)

    for padding_idx in surprisal.original_sentence.padding_idxs:
        insertion_points = tuple(
            filter(lambda t: t[1].idx == padding_idx, enumerate(new_tokens))
        )
        insertion_idx, _ = insertion_points[-1]
        new_tokens = (
            new_tokens[:insertion_idx]
            + [Token(text=PADDING, surprisal=np.nan, idx=padding_idx)]
            + new_tokens[insertion_idx:]
        )

    return dataclasses.replace(surprisal, tokens=tuple(new_tokens))


def get_surprisals_from_docker_container(
    model: str, container_id: str, sentences: tuple[Sentence, ...]
) -> tuple[SentenceSurprisal, ...]:
    sentence_strings = tuple(str(x) for x in sentences)
    logger.debug(f"Sending '{sentence_strings}' to container {container_id}")
    output = subprocess.check_output(
        [
            "docker",
            "exec",
            "-i",
            container_id,
            "get_surprisals",
            "/dev/stdin",
        ],
        input=("\n".join(sentence_strings) + "\n").encode("utf-8"),
        stderr=subprocess.DEVNULL,
    ).decode("utf-8")

    return _csv_output_to_surprisals(output, model, sentences)


def _align_token_regions_with_original_sentence(
    surprisal: SentenceSurprisal,
) -> SentenceSurprisal:
    orig_token_idx_to_is_critical = {
        i: surprisal.original_sentence.preprocessed_tokens[i].critical_region
        for i in range(len(surprisal.original_sentence.preprocessed_tokens))
    }
    new_tokens = []
    for surprisal_token in surprisal.tokens:
        new_tokens.append(
            dataclasses.replace(
                surprisal_token,
                critical_region=orig_token_idx_to_is_critical[surprisal_token.idx],
            )
        )

    return dataclasses.replace(surprisal, tokens=tuple(new_tokens))


def _cache_key(sentence: Sentence, model: str) -> str:
    return str((tuple(token.text for token in sentence.preprocessed_tokens), model))


def _remove_from_cache(s: str, model):
    key = str((tokenize(s), model))
    _SURPRISAL_CACHE.delete(key)


def remove_model_cache(model):
    query = f"*'{model}'*"
    num_keys = len(_SURPRISAL_CACHE.keys(query))
    response = input(
        f"This will delete {num_keys} entries from cache. Are you sure? Y/n\n"
    )
    if response != "Y":
        return
    total_deleted = 0
    for key in tqdm(_SURPRISAL_CACHE.keys(query)):
        _SURPRISAL_CACHE.delete(key)
        total_deleted += 1
    logger.info(f"Deleted {total_deleted} keys for model {model}")


def remove_from_cache(sentences, models):
    for s in sentences:
        for m in models:
            _remove_from_cache(s, m)


def store_in_cache(model: str, sentence_surprisal: SentenceSurprisal):
    surprisal_tokens = tuple(x.text for x in sentence_surprisal.tokens)
    original_tokens = tuple(
        x.text for x in sentence_surprisal.original_sentence.preprocessed_tokens
    )
    if surprisal_tokens != original_tokens:
        logger.warning(
            f"Cache token mismatch:\n\t{original_tokens}\n\t{surprisal_tokens}"
        )

    _SURPRISAL_CACHE[
        _cache_key(sentence_surprisal.original_sentence, model)
    ] = sentence_surprisal.to_json()


def get_surprisal_from_cache(
    sentence: Sentence, model: str
) -> Optional[SentenceSurprisal]:
    cached_json = _SURPRISAL_CACHE.get(_cache_key(sentence, model))
    if cached_json is None:
        return None
    surprisal = SentenceSurprisal.from_json(cached_json)
    # Replacing `original_sentence` since current `sentence` may contain different paddings.
    return dataclasses.replace(surprisal, original_sentence=sentence)


def _get_surprisals_from_cache(sentences: tuple[Sentence, ...], model: str):
    # TODO: only cache the raw model outputs, not SentenceSurprisal objects.
    non_cached_sentences_to_idx = {}
    cached_sentences = {}  # {idx in original list: SentenceSurprisal}
    for i, sentence in enumerate(sentences):
        cached = get_surprisal_from_cache(sentence, model)
        if cached:
            cached_sentences[i] = cached
        else:
            non_cached_sentences_to_idx[sentence] = i

    logger.debug(f"Found {len(cached_sentences)}/{len(sentences)} sentences in cache.")
    return cached_sentences, non_cached_sentences_to_idx


def get_model_base_and_checkpoint(model: str) -> tuple[str, str]:
    if "__" in model:
        model_base, checkpoint = model.split("__", maxsplit=1)
    else:
        model_base, checkpoint = model, None
    return model_base, checkpoint


def _tokenize_for_yedetore_models(sentence: Sentence) -> tuple[Token, ...]:
    tokens = []
    for original_token_idx, original_token in enumerate(sentence.preprocessed_tokens):
        subwords = childes_vocab.split_possesives_and_contractions(
            original_token.text
        ).split(" ")
        for subword in subwords:
            tokens.append(
                Token(
                    text=subword,
                    idx=original_token_idx,
                    critical_region=original_token.critical_region,
                )
            )
    return tuple(tokens)


def _feed_yedetore_get_probabs(net, input_word_ids, is_transformer, dictionary, device):
    # Returns shape (batch, length, vocab).
    net.eval()
    with torch.no_grad():
        if is_transformer:
            outputs = net(input_word_ids)
        else:
            hidden = net.init_hidden(bsz=1)
            outputs, _ = net(input_word_ids, hidden)

    # Back to (batch, length, vocab) for intuitiveness.
    outputs = outputs.transpose(0, 1)
    return torch.softmax(outputs, dim=-1)


def _get_inputs_for_childes_model(input_tokens, dictionary, device):
    # Yedetore models expect shape (length, batch, vocab).
    length = len(input_tokens)
    input_word_ids = torch.zeros(length, 1, dtype=torch.long)

    for i, token in enumerate(input_tokens):
        if token.text in dictionary.word2idx:
            word_id = dictionary.word2idx[token.text]
        else:
            logger.warning(f"UNK in sentence: {token.text}")
            word_id = dictionary.word2idx["<unk>"]

        input_word_ids[i, 0] = word_id

    input_word_ids = input_word_ids.to(device)
    return input_word_ids


def _get_surprisal_tokens_from_yedetore_model(
    net, input_tokens: tuple[Token, ...], dictionary, device, is_transformer
) -> tuple[Token, ...]:
    input_word_ids = _get_inputs_for_childes_model(input_tokens, dictionary, device)
    probabs = _feed_yedetore_get_probabs(
        net=net,
        input_word_ids=input_word_ids,
        is_transformer=is_transformer,
        dictionary=dictionary,
        device=device,
    )

    # Back to (batch, length, vocab) for intuitiveness.
    input_word_ids = input_word_ids.transpose(0, 1)

    next_word_probabs = torch.take_along_dim(
        probabs[:, :-1], input_word_ids[:, 1:].unsqueeze(-1), dim=-1
    )
    next_word_surprisals = -torch.log(next_word_probabs)
    first_word_zero_surprisals = torch.zeros((next_word_probabs.shape[0], 1, 1))

    surprisals = torch.concat(
        [first_word_zero_surprisals, next_word_surprisals],
        dim=1,
    )

    surprisal_tokens = []
    for i, token in enumerate(input_tokens):
        if token.text not in dictionary.word2idx:
            text = "<unk>"
        else:
            text = token.text
        surprisal_tokens.append(
            Token(text=text, surprisal=surprisals[0, i, 0].item(), idx=token.idx)
        )

    return tuple(surprisal_tokens)


def _load_childes_lstm(model_name, device):
    model_id = model_name.split("_")[-1]
    if len(model_id) == 1:
        model_id = "0" + model_id

    return torch.load(
        f"../models/yedetore/lstm/2-800-10-20-0.4-10{model_id}-LSTM-model.pt",
        map_location=device,
    )


def _load_childes_model(model_name, device):
    if "lstm" in model_name:
        return _load_childes_lstm(model_name, device)
    if "transformer" in model_name:
        return _load_childes_transformer(model_name, device)


def _load_childes_transformer(model_name, device):
    import model as yedetore_model

    model_id = model_name.split("_")[-1]
    if len(model_id) == 1:
        model_id = "0" + model_id

    # TODO: decide which seed to use out of their 10.
    m_state_dict = torch.load(
        f"../models/yedetore/transformer/04-500-800-10-0.2-5.0-04-10{model_id}-Transformer-state-dict.pt",
        map_location=device,
    )

    m = yedetore_model.TransformerModel(
        ntoken=17094, ninp=800, nhead=4, nhid=800, nlayers=4
    )
    m.load_state_dict(m_state_dict)
    return m


def _get_surprisals_from_retrained_transformer(
    sentences: tuple[Sentence, ...], model_name: str
):
    _, checkpoint = get_model_base_and_checkpoint(model_name)
    model_path = RETRAINING_DATA_PATH / "checkpoints" / f"{checkpoint}.pt"

    experiment_id = checkpoint

    from modified_external_sources.colorlessgreenRNNs.src.language_models import (
        dictionary_corpus as grnn_dictionary_corpus,
    )

    device = utils.get_device()

    if model_name in _TORCH_MODELS_CACHE:
        torch_model = _TORCH_MODELS_CACHE.get(model_name)
    else:
        torch_model = torch.load(model_path, map_location=device)
        _TORCH_MODELS_CACHE[model_name] = torch_model

    # Legacy.
    if "childes" in experiment_id:
        dataset_name = "childes"
    elif "wikipedia" in experiment_id:
        dataset_name = "wikipedia"
    else:
        raise ValueError(experiment_id)

    corpus_cache_key = (experiment_id, dataset_name)
    if corpus_cache_key in _CORPUS_CACHE:
        corpus = _CORPUS_CACHE[corpus_cache_key]
    else:
        corpus = grnn_dictionary_corpus.Corpus(
            base_path=RETRAINING_DATA_PATH,
            experiment_id=experiment_id,
            dataset_name=dataset_name,
        )
        _CORPUS_CACHE[corpus_cache_key] = corpus

    dictionary = corpus.dictionary

    return _get_surprisals_from_yedetore_model(
        sentences=sentences,
        net=torch_model,
        dictionary=dictionary,
        device=device,
        model_name=model_name,
        is_transformer=True,
    )


def _get_surprisals_from_yedetore_model(
    sentences,
    net,
    dictionary,
    device,
    model_name,
    is_transformer,
) -> tuple[SentenceSurprisal, ...]:
    sentence_token_lists = tuple(_tokenize_for_yedetore_models(s) for s in sentences)

    net.eval()
    with torch.no_grad():
        results = []
        for token_list, original_sentence in zip(sentence_token_lists, sentences):
            # TODO: use batches.
            surprisal_tokens = _get_surprisal_tokens_from_yedetore_model(
                net,
                token_list,
                dictionary,
                device,
                is_transformer=is_transformer,
            )
            results.append(
                SentenceSurprisal(
                    tokens=surprisal_tokens,
                    model=model_name,
                    original_sentence=original_sentence,
                )
            )

    return tuple(results)


def _get_surprisals_from_childes_model(
    sentences: tuple[Sentence, ...], model: str
) -> tuple[SentenceSurprisal, ...]:
    device = utils.get_device()

    net = _load_childes_model(model, device)
    dictionary = childes_vocab.Dictionary()
    sentence_token_lists = tuple(_tokenize_for_yedetore_models(s) for s in sentences)

    return _get_surprisals_from_yedetore_model(
        sentences=sentences,
        net=net,
        dictionary=dictionary,
        device=device,
        model_name=model,
        is_transformer="transformer" in model,
    )


def _get_bert_masked_surprisal(
    sentence: Sentence, bert_pipeline, bert_model: str
) -> SentenceSurprisal:
    critical_tokens = [
        (i, x) for i, x in enumerate(sentence.preprocessed_tokens) if x.critical_region
    ]
    assert (
        len(critical_tokens) == 1
    ), f"BERT surprisals can only be measured at single masked token"

    target = critical_tokens[0][1].text
    target_token_idx = critical_tokens[0][0]

    assert target in bert_pipeline.tokenizer.vocab, f"{target} not in BERT vocabulary."

    masked_str = " ".join(
        [
            token.text if not token.critical_region else "[MASK]"
            for token in sentence.preprocessed_tokens
        ]
    )
    result = bert_pipeline(masked_str, targets=target)[0]
    assert result["token_str"] == target
    probab = result["score"]
    surprisal = -np.log2(probab).item()

    tokens = []
    for i, original_token in enumerate(sentence.preprocessed_tokens):
        if i == target_token_idx:
            token_surprisal = surprisal
        else:
            token_surprisal = 0.0
        tokens.append(dataclasses.replace(original_token, surprisal=token_surprisal))

    sentence_surprisal = SentenceSurprisal(
        tokens=tuple(tokens), model=bert_model, original_sentence=sentence
    )
    return sentence_surprisal


def _get_surprisals_from_bert(
    sentences: tuple[Sentence, ...], bert_model: str
) -> tuple[SentenceSurprisal, ...]:
    bert_pipeline = transformers.pipeline("fill-mask", model=bert_model)
    return tuple(
        _get_bert_masked_surprisal(sentence, bert_pipeline, bert_model)
        for sentence in sentences
    )


def get_surprisals(
    sentences: tuple[Sentence, ...], model: str
) -> tuple[SentenceSurprisal, ...]:
    (
        cached_sentences,
        non_cached_sentences_to_idx,
    ) = _get_surprisals_from_cache(sentences, model)

    non_cached_sentences = tuple(non_cached_sentences_to_idx)
    results_from_live_model = {}

    if non_cached_sentences:
        if model == GPT3:  # Shorthand.
            model = f"{GPT3}__{GPT3_DEFAULT_ENGINE}"
        if model.startswith(f"{GPT3}__"):
            _, engine = model.split("__")

            if os.getenv("POS_LM_DEV_MACHINE"):
                logger.warning(
                    f"Dev machine, using cheaper GPT-3 model {_GPT3_CHEAP_ENGINE}."
                )
                engine = _GPT3_CHEAP_ENGINE

            surprisals = _get_surprisals_from_openai(
                non_cached_sentences, engine=engine
            )
        elif model.startswith("retrained__"):
            surprisals = _get_surprisals_from_retrained_transformer(
                non_cached_sentences, model
            )
        elif model.startswith("childes_"):
            surprisals = _get_surprisals_from_childes_model(non_cached_sentences, model)
        elif model.startswith("bert"):
            surprisals = _get_surprisals_from_bert(
                non_cached_sentences, bert_model=model
            )
        else:
            surprisals = get_surprisals_from_docker_container(
                model=model,
                container_id=_start_or_get_container(model),
                sentences=non_cached_sentences,
            )

        for sentence_surprisal in surprisals:
            results_from_live_model[
                non_cached_sentences_to_idx[sentence_surprisal.original_sentence]
            ] = sentence_surprisal

    results = []
    for i, sentence in enumerate(sentences):
        if i in cached_sentences:
            results.append(cached_sentences[i])
        else:
            sentence_surprisal = results_from_live_model[i]
            store_in_cache(model, sentence_surprisal)
            results.append(sentence_surprisal)

    return tuple(map(_align_token_regions_with_original_sentence, results))


def normalize_training_file_string(s):
    s = re.sub(r" <eos>\n$", "", s)
    return re.sub(r" 's", "'s", s)


def load_grammars_toml() -> dict:
    with open("grammars.toml", "r") as f:
        return toml.load(f)


def load_grammars(
    names_size, retraining_corpus=False
) -> dict[str, dict[str, tuple[nltk.CFG, ...]]]:
    # {"ATB": {"FG": (grammar1, grammar2, ...), "XG": ..., "FX": ..., "XX: ... }, "PG": {...}, ... }
    grammars = collections.defaultdict(lambda: collections.defaultdict(list))
    config = load_grammars_toml()

    if retraining_corpus:
        grammar_type = "RETRAINING_GRAMMARS"
    else:
        grammar_type = "GRAMMARS"

    base_s_derivations = config["BASE_S_DERIVATIONS"]
    base_grammar_str = config["BASE_GRAMMAR"]
    names_grammar_str = config[f"NAMES_{names_size.upper()}"]

    for p, phenomenon in enumerate(config["PHENOMENA"]):
        phenomenon_grammars = config[phenomenon][grammar_type]

        for g, grammar_str in enumerate(phenomenon_grammars):
            for condition, s_derivation_str in base_s_derivations.items():
                full_grammar_str = (
                    s_derivation_str
                    + "\n"
                    + base_grammar_str
                    + "\n"
                    + names_grammar_str
                    + "\n"
                    + grammar_str
                )
                grammar = nltk.CFG.fromstring(full_grammar_str)

                grammars[phenomenon][condition].append(grammar)

    ret = {}
    for x, y in grammars.items():
        ret[x] = {y: tuple(z) for y, z in grammars[x].items()}

    return ret


def _break_cfg_terminals_into_single_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    # Rules such as A -> "w1 w2" yield a tuple `('w1 w2',)`, break it into `('w1', 'w2',)`.
    single_words = ()
    for t in tokens:
        single_words += tokenize(t)
    return single_words


def _merge_possessives(tokens: tuple[str, ...]) -> tuple[str, ...]:
    new_tokens = []

    if not isinstance(tokens, tuple):
        tokens = tuple(tokens)

    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i + 1] == "'s":
            new_tokens.append("".join(tokens[i : i + 2]))
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)


def _get_terminals(tree):
    curr_terminals = []
    for daughter in tree:
        if isinstance(daughter, nltk.Tree):
            curr_terminals += _get_terminals(daughter)
        else:
            curr_terminals += [(tree.label(), daughter)]

    return curr_terminals


def grammar_output_to_sentence(tokens: tuple[str, ...]) -> Sentence:
    tokens = _break_cfg_terminals_into_single_tokens(tokens)
    tokens = _merge_possessives(tokens)
    return tokens_to_sentence(tokens)


def get_lexical_choices(
    output_tokens: tuple[str, ...], grammar: nltk.CFG
) -> frozenset[tuple[str, str]]:
    # Ugly way to get derivation choices which NLTK doesn't provide easily. Assuming grammar is unambiguous.

    parser = earleychart.EarleyChartParser(grammar)
    trees = tuple(parser.parse(output_tokens))
    assert len(trees) == 1

    terminals = _get_terminals(trees[0])

    category_to_terminal = {}

    for category, terminal in terminals:
        if len(grammar._lhs_index[nltk.Nonterminal(category)]) < 2:
            continue
        category_to_terminal[category] = terminal

    return frozenset(category_to_terminal.items())


def generate_sentences_from_cfg(grammar: nltk.CFG) -> Iterable[Sentence]:
    for grammar_output in generate.generate(grammar):
        yield grammar_output_to_sentence(grammar_output)


def iterate_phenomenon_sentences(
    phenomenon, names_size, conditions=None, retraining_corpus=False
):
    phenomenon_to_grammars = load_grammars(
        names_size, retraining_corpus=retraining_corpus
    )
    for curr_condition, curr_grammars in phenomenon_to_grammars[phenomenon].items():
        if conditions is not None:
            if curr_condition not in conditions:
                continue

        for grammar in curr_grammars:
            for sentence in generate_sentences_from_cfg(grammar):
                yield sentence
