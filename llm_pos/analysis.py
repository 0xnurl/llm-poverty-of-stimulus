import collections
import itertools
import operator
import pathlib
import pickle
import re

import pandas as pd
import tqdm

import lib

_INTERACTION_TITLES = {
    "interaction": "$\Delta_{-filler} - \Delta_{+filler}$",
    "delta_plus_filler": "$\Delta_{+filler}$",
    "delta_minus_filler": "$\Delta_{-filler}$",
}


def _calculate_subset_to_superset_mapping(
    col1, col2, max_diff
) -> dict[tuple[int, frozenset], frozenset[tuple[int, frozenset]]]:
    col1_to_col2_supersets = collections.defaultdict(list)
    for c1, c2 in tqdm.tqdm(itertools.product(col1, col2), total=len(col1) * len(col2)):
        c1_set = c1[1]
        c2_set = c2[1]
        if len(c2_set) - len(c1_set) <= max_diff and c1_set.issubset(c2_set):
            col1_to_col2_supersets[c1].append(c2)

    return {x: frozenset(y) for x, y in col1_to_col2_supersets.items()}


def gen_interaction_dataframe(surprisal_df_id):
    interaction_path = lib.DATA_PATH / f"{surprisal_df_id}_full_interaction.pickle"
    if pathlib.Path(interaction_path).exists():
        with open(interaction_path, "rb") as f:
            return pickle.load(f)

    df = lib.load_surprisal_dataframe(surprisal_df_id)

    num_grammars = df.grammar.max() + 1
    tuples = []

    for g in range(num_grammars):
        grammar_df = df[df.grammar == g]

        pfpg_rows = grammar_df[grammar_df.condition == lib.PLUS_FILLER_PLUS_GAP]
        mfpg_rows = grammar_df[grammar_df.condition == lib.MINUS_FILLER_PLUS_GAP]
        pfmg_rows = grammar_df[grammar_df.condition == lib.PLUS_FILLER_MINUS_GAP]
        mfmg_rows = grammar_df[grammar_df.condition == lib.MINUS_FILLER_MINUS_GAP]

        pfpg_lex_choices, mfpg_lex_choices, pfmg_lex_choices, mfmg_lex_choices = tuple(
            map(
                lambda condition_rows: tuple(
                    map(frozenset, condition_rows.lexical_choices)
                ),
                (pfpg_rows, mfpg_rows, pfmg_rows, mfmg_rows),
            )
        )

        pfpg = tuple(zip(pfpg_rows.index, pfpg_lex_choices))
        mfpg = tuple(zip(mfpg_rows.index, mfpg_lex_choices))
        pfmg = tuple(zip(pfmg_rows.index, pfmg_lex_choices))
        mfmg = tuple(zip(mfmg_rows.index, mfmg_lex_choices))

        pfpg_to_mfpg_supersets = _calculate_subset_to_superset_mapping(
            pfpg, mfpg, max_diff=1
        )
        pfpg_to_pfmg_supersets = _calculate_subset_to_superset_mapping(
            pfpg, pfmg, max_diff=1
        )
        mfpg_to_mfmg_supersets = _calculate_subset_to_superset_mapping(
            mfpg, mfmg, max_diff=1
        )
        pfmg_to_mfmg_supersets = _calculate_subset_to_superset_mapping(
            pfmg, mfmg, max_diff=2
        )

        col_2_3_to_col4 = {}
        for c2, c3 in tqdm.tqdm(
            itertools.product(
                mfpg_to_mfmg_supersets.keys(), pfmg_to_mfmg_supersets.keys()
            ),
            total=len(mfpg_to_mfmg_supersets) * len(pfmg_to_mfmg_supersets),
        ):
            col_2_3_to_col4[(c2, c3)] = (
                mfpg_to_mfmg_supersets[c2] & pfmg_to_mfmg_supersets[c3]
            )

        for c1 in pfpg_to_mfpg_supersets:
            for c2 in pfpg_to_mfpg_supersets[c1]:
                for c3 in pfpg_to_pfmg_supersets[c1]:
                    for c4 in col_2_3_to_col4[(c2, c3)]:
                        tuples.append((c1, c2, c3, c4))

    df_dict = df.to_dict()

    result_rows = []
    for tup in tqdm.tqdm(tuples):
        idxs = tuple(x[0] for x in tup)
        i1, i2, i3, i4 = idxs

        # Success criterion for net:
        #    ([+F,+G] - [+F,-G]) <? ([-F,-G] - [-F,+G])
        # => ([+F,+G] - [-F,+G]) - ([+F,-G] - [-F,-G]) <? 0
        interaction = (
            df_dict["critical_surprisal_average"][i1]
            - df_dict["critical_surprisal_average"][i2]
        ) - (
            df_dict["critical_surprisal_average"][i3]
            - df_dict["critical_surprisal_average"][i4]
        )

        condition_to_sentence = {
            df_dict["condition"][i]: df_dict["sentence"][i] for i in idxs
        }
        condition_to_surprisal = {
            f"{df_dict['condition'][i]} surprisal": df["critical_surprisal_average"][i]
            for i in idxs
        }
        result_rows.append(
            {
                **condition_to_sentence,
                **condition_to_surprisal,
                "delta_plus_filler": condition_to_surprisal[
                    f"{lib.PLUS_FILLER_PLUS_GAP} surprisal"
                ]
                - condition_to_surprisal[f"{lib.PLUS_FILLER_MINUS_GAP} surprisal"],
                "delta_minus_filler": condition_to_surprisal[
                    f"{lib.MINUS_FILLER_PLUS_GAP} surprisal"
                ]
                - condition_to_surprisal[f"{lib.MINUS_FILLER_MINUS_GAP} surprisal"],
                "interaction": interaction,
                "original_idxs": idxs,
            }
        )

    interaction_df = pd.DataFrame(result_rows)
    interaction_df.reset_index(inplace=True)
    interaction_df.to_csv(lib.DATA_PATH / f"{surprisal_df_id}_full_interaction.csv")
    with open(interaction_path, "wb") as f:
        pickle.dump(interaction_df, f)
    return interaction_df


def _scatter_interaction_2d(
    ax, df, model, min_x, max_x, min_y, max_y, color, label, model_label
):
    model_base, checkpoint = lib.get_model_base_and_checkpoint(model)

    title = r"{\normalsize %s}" % model_base.upper()
    if model_label is None:
        if checkpoint is None:
            model_label = "original"
        else:
            model_label = checkpoint.replace("_", "-")
    title += "\n" + r"{\small %s}" % model_label
    if label:
        title += "\n" + r"{\small %s}" % label
    ax.set_title(title)

    x = df["delta_minus_filler"]
    y = df["delta_plus_filler"]
    interaction = df.interaction

    below_zero = interaction[interaction < 0.0]
    below_zero_ratio = len(below_zero) / len(interaction)

    x_label = r"$%s > 0$ (%s / %s)" % (
        f"{below_zero_ratio:.2f}",
        f"{len(below_zero):,}",
        f"{len(interaction):,}",
    )
    if label:
        x_label += f"\n {label}"

    ax.set_ylabel(r"$\Delta{+filler}$")
    ax.set_xlabel(r"$\Delta{-filler}$")

    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))

    ax.hlines(y=0, xmin=min_x, xmax=max_x, colors="gray", linestyles="-", lw=1)
    ax.vlines(x=0, ymin=min_y, ymax=max_y, colors="gray", linestyles="-", lw=1)
    ax.axline((0, 0), slope=1, color="gray", lw=1)

    ax.scatter(
        x,
        y,
        color=color,
        s=2,
        alpha=0.3,
    )


def _scatter_interaction(ax, x, y, model, color, min_val, max_val, label, model_label):
    model_base, checkpoint = lib.get_model_base_and_checkpoint(model)
    title = r"{\normalsize %s}" % model_base.upper()
    if model_label is None:
        if checkpoint is None:
            model_label = "original"
        else:
            model_label = checkpoint.replace("_", "-")
    title += "\n" + r"{\small %s}" % model_label
    ax.set_title(title)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.hlines(y=0, xmin=x.min(), xmax=x.max(), colors="purple", linestyles="-", lw=1.5)
    ax.set_ylim((min_val, max_val))

    above_zero = y[y > 0.0]
    above_zero_ratio = len(above_zero) / len(y)
    x_label = r"$%s > 0$ (%s / %s)" % (
        f"{above_zero_ratio:.2f}",
        f"{len(above_zero):,}",
        f"{len(y):,}",
    )
    if label:
        x_label += f"\n {label}"
    ax.set_xlabel(x_label)

    ax.scatter(
        x,
        y,
        c=color,
        s=2,
        alpha=0.4,
    )


def _get_interaction_df(model, phenomenon, df_id, interaction_column):
    surprisal_df_id = f"{phenomenon}__model__{model}__cfg_{df_id}"
    return gen_interaction_dataframe(surprisal_df_id)


def get_model_accuracy(model, phenomenon, df_id, interaction_column):
    model_df = _get_interaction_df(model, phenomenon, df_id, interaction_column)
    interaction = model_df[interaction_column]

    if interaction_column in {"interaction", "delta_plus_filler"}:
        op = operator.lt
    elif interaction == "delta_minus_filler":
        op = operator.ge
    else:
        raise ValueError(interaction_column)

    above_zero_rows = interaction[op(interaction, 0.0)]
    return len(above_zero_rows) / len(interaction)


def get_worst_sentence_tuples(surprisal_df_id, n, unique_only):
    df = gen_interaction_dataframe(surprisal_df_id)
    largest_value_idxs = df.interaction.argsort()[::-1]

    seen_sentences = set()
    worst_rows = []

    for i in largest_value_idxs:
        curr_row = df.iloc[i]
        sentence = curr_row[lib.PLUS_FILLER_PLUS_GAP]

        if not unique_only or (sentence not in seen_sentences):
            seen_sentences.add(sentence)
            worst_rows.append(curr_row)
            if len(worst_rows) == n:
                break

    return pd.DataFrame(worst_rows)


def print_worst_sentences(surprisal_df_id, n, unique_only):
    worst_rows = get_worst_sentence_tuples(surprisal_df_id, n, unique_only)
    rows = worst_rows[["interaction", lib.PLUS_FILLER_PLUS_GAP]]

    for i in range(len(rows)):
        row = rows.iloc[i]
        print(f"{row[lib.PLUS_FILLER_PLUS_GAP]}\t{row.interaction:.2f}")


def worst_failures_to_latex(surprisal_df_id, n, unique_only, model_labels={}):
    worst_tuples_df = get_worst_sentence_tuples(surprisal_df_id, n, unique_only)
    surprisal_df = lib.load_surprisal_dataframe(surprisal_df_id)

    model = surprisal_df.iloc[0].model

    model_label = model_labels.get(model, model.upper())
    phenomenon = surprisal_df.iloc[0].phenomenon

    text = r"""\subsection{%s -- %s}
""" % (
        phenomenon,
        model_label,
    )

    for i, row in enumerate(worst_tuples_df.iterrows()):
        sentences = row[1][list(lib.CONDITIONS)]
        original_df_idxs = row[1].original_idxs
        original_df_rows = surprisal_df.iloc[list(original_df_idxs)]
        critical_words = [x[0] for x in original_df_rows.critical_region]
        surprisals = [x[0] for x in original_df_rows.critical_surprisal_values]

        formatted_sentences = []
        for sentence, critical_word, surprisal in zip(
            sentences, critical_words, surprisals
        ):
            surprisal_str = f"({surprisal:.2f})"
            replace_regex = r"\s(" + critical_word + ")"
            formatted_sentence = re.sub(
                replace_regex, r" \\textbf{\1 " + surprisal_str + "}", sentence
            )
            formatted_sentence = formatted_sentence.replace(
                "I know who ", r"I know \underline{who} "
            )
            formatted_sentence = formatted_sentence.replace(
                "I know that ", r"I know \underline{that} "
            )

            formatted_sentences.append(formatted_sentence)

        pfpg, mfpg, pfmg, mfmg = formatted_sentences

        table_text = r"""
\begin{exe}
\ex{
\begin{tabularx}{\linewidth}[]{|l|L|L|}
\hline
& $+gap$ & $-gap$ \\ \hline
$+filler$ & """
        table_text += f"{pfpg} & *{pfmg}"
        table_text += r"""
\\ \hline
$-filler$ & """
        table_text += f"*{mfpg} & {mfmg}"
        table_text += r"""\\ 
\hline
\multicolumn{3}{|l|}{"""
        table_text += (
            "$\Delta_{-filler} - \Delta_{+filler} = "
            + f"{row[1].interaction * -1:.2f}$"
        )
        table_text += r"""}
\\
\hline
\end{tabularx}
\label{table:failure:%s:%s:%s}}
\end{exe} \\
""" % (
            phenomenon.lower(),
            model.replace("_", "-").lower(),
            i,
        )

        text += table_text

    return text


def worst_failures_to_latex_multiple(
    surprisal_df_ids, n, unique_only, filename, model_labels={}
):
    text = ""
    for df_id in surprisal_df_ids:
        text += worst_failures_to_latex(df_id, n, unique_only, model_labels) + "\n\n"

    with open(lib.DATA_PATH / filename, "w") as f:
        f.write(text)
