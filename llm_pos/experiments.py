import collections
import itertools
import math
import pickle
import random
from typing import Optional

import nltk
import numpy as np
import pandas as pd
import tqdm
from loguru import logger
from nltk.parse import generate

import analysis
import lib

_BATCH_SIZE = 100


def _run_experiment(
    phenomenon,
    condition,
    grammars,
    model,
    plus_g_plus_f_lexical_choices: Optional[frozenset[frozenset[tuple[str, str]]]],
) -> pd.DataFrame:
    result_rows = []

    for g, grammar in enumerate(grammars):
        logger.info(f"Grammar {g+1}/{len(grammars)}")
        grammar_outputs = tuple(generate.generate(grammar))
        grammar_lexical_choices = tuple(
            lib.get_lexical_choices(s, grammar) for s in grammar_outputs
        )

        if plus_g_plus_f_lexical_choices:
            filtered_idxs = []
            for i, curr_grammar_lex_choices in enumerate(grammar_lexical_choices):
                for curr_plus_g_plus_f_lexical_choices in plus_g_plus_f_lexical_choices:
                    if curr_plus_g_plus_f_lexical_choices.issubset(
                        curr_grammar_lex_choices
                    ):
                        filtered_idxs.append(i)
                        break

            grammar_outputs = tuple(grammar_outputs[i] for i in filtered_idxs)
            grammar_lexical_choices = tuple(
                grammar_lexical_choices[i] for i in filtered_idxs
            )

        progress_bar = tqdm.tqdm(total=len(grammar_outputs))

        sentences = tuple(lib.grammar_output_to_sentence(x) for x in grammar_outputs)
        assert len(sentences) == len(set(sentences))

        for batch_start in range(0, len(sentences), _BATCH_SIZE):
            batch_sentences = sentences[batch_start : batch_start + _BATCH_SIZE]

            batch_surprisals = lib.get_surprisals(batch_sentences, model)
            for i, surprisal in enumerate(batch_surprisals):
                critical_surprisal_values = lib.get_critical_surprisals(surprisal)
                result_rows.append(
                    {
                        "model": model,
                        "phenomenon": phenomenon,
                        "condition": condition,
                        "grammar": g,
                        "sentence": str(surprisal.original_sentence),
                        "critical_region": lib.get_critical_words(
                            surprisal.original_sentence
                        ),
                        "critical_surprisal_values": critical_surprisal_values,
                        "critical_surprisal_average": np.mean(
                            critical_surprisal_values
                        ),
                        "lexical_choices": tuple(
                            sorted(grammar_lexical_choices[batch_start + i])
                        ),
                    }
                )
                progress_bar.update(1)

    return pd.DataFrame(result_rows)


def _generate_train_test_sentences_from_grammar(
    grammar: nltk.CFG,
    train_ratio: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    all_sentences = []
    # {('ADJ', 'good') -> {1, 17}}
    single_lex_choice_to_sentence_idxs = collections.defaultdict(set)
    full_lex_choices_to_sentence_idx = {}
    # {'ADJ' -> {'good', 'bad'}}
    all_lex_choices_per_category = collections.defaultdict(set)

    for i, tokens in enumerate(generate.generate(grammar)):
        lex_choices = lib.get_lexical_choices(tokens, grammar)
        full_lex_choices_to_sentence_idx[lex_choices] = i
        for category, lex_choice in lex_choices:
            all_lex_choices_per_category[category].add(lex_choice)
            single_lex_choice_to_sentence_idxs[(category, lex_choice)].add(i)

        all_sentences.append(str(lib.grammar_output_to_sentence(tokens)))

    test_lex_choices = {}

    for category, lex_choices in all_lex_choices_per_category.items():
        num_test_lex_choices = math.ceil((1 - train_ratio) * len(lex_choices))
        category_test_lex_choices = tuple(
            random.sample(sorted(lex_choices), num_test_lex_choices)
        )
        test_lex_choices[category] = category_test_lex_choices

    all_test_lex_choices = tuple(
        map(
            frozenset,
            itertools.product(
                *[
                    [(cat, lex_choice) for lex_choice in test_lex_choices[cat]]
                    for cat in test_lex_choices
                ]
            ),
        )
    )
    test_idxs = set()
    for test_lex_choice in all_test_lex_choices:
        test_idxs.add(full_lex_choices_to_sentence_idx[test_lex_choice])

    test_lex_choice_pairs = []
    for (cat1, cat1_choices), (cat2, cat2_choices) in itertools.combinations(
        test_lex_choices.items(), 2
    ):
        for lex_choice1, lex_choice2 in itertools.product(cat1_choices, cat2_choices):
            test_lex_choice_pairs.append(((cat1, lex_choice1), (cat2, lex_choice2)))

    training_idxs = set(range(len(all_sentences)))

    for (cat1, lex_choice1), (cat2, lex_choice2) in test_lex_choice_pairs:
        sentences1 = single_lex_choice_to_sentence_idxs[(cat1, lex_choice1)]
        sentences2 = single_lex_choice_to_sentence_idxs[(cat2, lex_choice2)]
        training_idxs -= sentences1 & sentences2

    training_sentences = tuple(all_sentences[i] for i in sorted(training_idxs))
    test_sentences = tuple(all_sentences[i] for i in sorted(test_idxs))

    return training_sentences, test_sentences


def generate_training_test_data(
    phenomenon: str,
    train_ratio: float,
    names_size: str,
    grammar_idx: Optional[int] = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Generate only [+f,+g] sentences.
    # Returns (train set, test set).
    grammars = lib.load_grammars(names_size=names_size)[phenomenon][
        lib.PLUS_FILLER_PLUS_GAP
    ]
    if grammar_idx is not None:
        grammars = (grammars[grammar_idx],)

    train_sentences = []
    test_sentences = []
    for grammar in grammars:
        (
            grammar_train_sentences,
            grammar_test_sentences,
        ) = _generate_train_test_sentences_from_grammar(grammar, train_ratio)
        train_sentences += grammar_train_sentences
        test_sentences += grammar_test_sentences

    return tuple(train_sentences), tuple(test_sentences)


def _get_plus_g_plus_f_lexical_choices_for_sentences(
    phenomenon: str,
    names_size: str,
    sentences_file: str,
) -> frozenset[frozenset[tuple[str, str]]]:
    plus_f_plus_g_grammars = lib.load_grammars(names_size)[phenomenon][
        lib.PLUS_FILLER_PLUS_GAP
    ]
    intersect_with_sentences = set()

    with (lib.RETRAINING_DATA_PATH / sentences_file).open("r") as f:
        for line in f.readlines():
            intersect_with_sentences.add(lib.normalize_training_file_string(line))

    plus_g_plus_f_lexical_choices = set()
    for plus_f_plus_g_grammar in plus_f_plus_g_grammars:
        for plus_f_plus_g_tokens in generate.generate(plus_f_plus_g_grammar):
            plus_f_plus_g_sentence = lib.grammar_output_to_sentence(
                plus_f_plus_g_tokens
            )
            if str(plus_f_plus_g_sentence) in intersect_with_sentences:
                lex_choices = lib.get_lexical_choices(
                    plus_f_plus_g_tokens, plus_f_plus_g_grammar
                )
                plus_g_plus_f_lexical_choices.add(lex_choices)

    return frozenset(plus_g_plus_f_lexical_choices)


def run_phenomenon(
    phenomenon: str,
    model: str,
    names_size: str,
    grammar_idx: Optional[int] = None,
    intersect_with_sentence_file: Optional[str] = None,
):
    phenomenon_condition_to_grammars = lib.load_grammars(names_size)[phenomenon]
    if grammar_idx is not None:
        phenomenon_condition_to_grammars = {
            condition: (grammars[grammar_idx],)
            for condition, grammars in phenomenon_condition_to_grammars.items()
        }

    model_df = pd.DataFrame()

    if intersect_with_sentence_file:
        intersect_with_lexical_choices = (
            _get_plus_g_plus_f_lexical_choices_for_sentences(
                phenomenon, names_size, intersect_with_sentence_file
            )
        )
    else:
        intersect_with_lexical_choices = None

    for condition, grammars in phenomenon_condition_to_grammars.items():
        logger.info(f"Running {phenomenon} condition {condition} on model {model}...")
        current_df = _run_experiment(
            grammars=grammars,
            phenomenon=phenomenon,
            condition=condition,
            model=model,
            plus_g_plus_f_lexical_choices=intersect_with_lexical_choices,
        )
        model_df = pd.concat([model_df, current_df])

    model_df.reset_index(inplace=True)

    filename = lib.get_surprisal_dataframe_id(
        phenomenon=phenomenon,
        model=model,
        names_size=names_size,
        grammar_version=lib.get_grammar_version(),
        sentence_file=intersect_with_sentence_file,
        grammar_idx=grammar_idx,
    )
    with open(lib.DATA_PATH / f"{filename}.pickle", "wb") as f:
        pickle.dump(model_df, f)

    model_df.to_csv(lib.DATA_PATH / f"{filename}.csv", float_format="%.5f")

    analysis.gen_interaction_dataframe(surprisal_df_id=filename)
