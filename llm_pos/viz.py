import math
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import analysis
import lib


def _ordered_set(x: Iterable) -> tuple:
    ordered = []
    set_ = set()
    for i in x:
        if i not in set_:
            set_.add(i)
            ordered.append(i)
    return tuple(ordered)


def _draw_surprisal_figure(
    ax,
    sentence_surprisals: tuple[lib.SentenceSurprisal, ...],
    max_y,
    draw_legend,
    labels,
    text_box,
    model_name,
):
    padded = lib.pad(sentence_surprisals)

    surprisal_vals = []
    for s in padded:
        curr_surprisals = []
        for t in s.tokens:
            curr_surprisals.append(t.surprisal)
        surprisal_vals.append(tuple(curr_surprisals))
    surprisal_vals = tuple(surprisal_vals)

    x_labels = []
    for i in range(len(padded[0].tokens)):
        idx_tokens = [s.tokens[i] for s in padded]
        token_texts = map(
            lambda t: t.text if t.text != lib.PADDING else "-", idx_tokens
        )

        label = "\n/".join(f"${t}$" for t in _ordered_set(token_texts))
        x_labels.append(label)

        if any(x.critical_region for x in idx_tokens):
            ax.axvspan(i - 0.25, i + 0.25, facecolor="gray", alpha=0.2, linewidth=2.1)

    x_idxs = tuple(range(len(padded[0].tokens)))

    for i, s in enumerate(surprisal_vals):
        ax.plot(x_idxs, s, label=f"S{i + 1}" if not labels else labels[i], alpha=0.8)

    ax.set_xticks(x_idxs)
    ax.set_xticklabels(x_labels, rotation="45", fontsize="large")
    ax.set_ylabel("Surprisal")
    if draw_legend:
        ax.legend(loc="upper left")

    if text_box:
        ax.text(
            1,
            0.05,
            text_box,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"boxstyle": "square", "facecolor": "#fff", "alpha": 0.8},
        )

    # ax.spines["top"].set_visible(False)
    ax.set_ylim((0, max_y))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title(model_name)


def plot_surprisals(
    sentences: tuple[str, ...],
    models: tuple[str, ...],
    model_to_label=None,
    labels=None,
    deltas=None,
    save_to=None,
    dpi=200,
):
    surprisal_per_model = lib.get_surprisals_per_model(sentences, models=models)
    surprisal_per_model = {x: lib.pad(y) for x, y in surprisal_per_model.items()}

    max_y = max(
        t.surprisal for m in models for s in surprisal_per_model[m] for t in s.tokens
    )

    plot_rows = math.ceil(len(models) / 2)
    plt.rcParams.update({"text.usetex": True})

    sns.set_palette(sns.color_palette())
    sns.set_style("darkgrid")

    fig, axes = plt.subplots(
        plot_rows,
        2,
        figsize=(10, 2 * plot_rows),
        dpi=dpi,
        constrained_layout=True,
    )

    if len(models) % 2 == 1:
        fig.delaxes(axes[-1, -1])

    for i, (ax, model) in enumerate(zip(axes.flatten(), models)):
        if not deltas:
            delta_label = None
        else:
            delta_label = (
                r"$\Delta_{-filler} - \Delta_{+filler} = "
                + f"{(deltas[i][0] - deltas[i][1]):.2f}$"
            )

        if model_to_label and model in model_to_label:
            model_name = model_to_label[model]
        else:
            model_name = model.upper()

        _draw_surprisal_figure(
            ax,
            surprisal_per_model[model],
            max_y=max_y,
            draw_legend=i == 0,
            labels=labels,
            text_box=delta_label,
            model_name=model_name,
        )

    if save_to:
        plt.savefig(save_to, format=save_to[-3:], dpi=dpi, bbox_inches="tight")

    fig.show()


def plot_accuracy_bars(
    phenomenon: str,
    models,
    df_id,
    interaction_column="interaction",
    model_to_label=None,
    figsize=(9, 3),
    dpi=140,
    save_to=None,
):
    rows = []

    if type(df_id) is str:
        df_ids = (df_id,) * len(models)
    elif type(df_id) is tuple:
        assert len(df_id) == len(models)
        df_ids = df_id
    else:
        raise ValueError(df_id)

    for model, model_df_id in zip(models, df_ids):
        accuracy = analysis.get_model_accuracy(
            model, phenomenon, model_df_id, interaction_column
        )
        if model_to_label and model in model_to_label:
            model_name = model_to_label[model]
        else:
            model_name = model.upper()
        rows.append({"model": model_name, "accuracy": accuracy * 100})

    df = pd.DataFrame(rows)

    sns.set_theme(rc={"figure.dpi": dpi, "figure.figsize": figsize})
    sns.set_style("darkgrid")
    sns.set_palette(sns.color_palette("hls", len(models)))
    ax = sns.barplot(data=df, x="model", y="accuracy")

    ax.axhline(50, color=(0.5, 0.5, 0.5, 0.3), zorder=-1, linewidth=2, linestyle="--")
    y_ticks = list(range(0, 110, 20))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y}%" for y in y_ticks])

    bar_labels = [f"{x:.1f}%" for x in df.accuracy]
    ax.bar_label(ax.containers[0], labels=bar_labels)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)

    title = f"{phenomenon} model accuracy\n"
    title += {
        "interaction": "$\Delta_{+filler} > \Delta_{-filler}$",
        "delta_plus_filler": "$\Delta_{+filler} > 0$",
        "delta_minus_filler": "$\Delta_{-filler}$",
    }[interaction_column]

    ax.set_title(title, fontsize=15)
    ax.set(ylim=(0.0, 100.0))

    if save_to:
        plt.savefig(save_to, format=save_to[-3:], dpi=dpi, bbox_inches="tight")

    plt.show()
