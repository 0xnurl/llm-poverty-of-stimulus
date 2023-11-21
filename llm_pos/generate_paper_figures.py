import analysis

_GRAMMAR_VERSION = "v2023-11-10a"

_GPT3 = "gpt3__text-davinci-003"
_TRANSFORMER_WIKIPEDIA = "retrained__PG__retraining__dataset_wikipedia__seed_1000__v2023-11-10a__tiny__samples_0__transformer__76a79487__epoch_18"

_MODELS = list(
    reversed(
        (
            _GPT3,
            "gpt-j",
            "gpt2",
            _TRANSFORMER_WIKIPEDIA,
            "grnn",
            "childes_transformer_8",
            "childes_lstm_8",
        )
    )
)

_MODEL_LABELS = {
    _GPT3: "GPT-3",
    "gpt-j": "GPT-j",
    "gpt2": "GPT-2",
    _TRANSFORMER_WIKIPEDIA: "Wikipedia Transformer",
    "grnn": "Wikipedia LSTM",
    "childes_transformer_8": "CHILDES Transformer",
    "childes_lstm_8": "CHILDES LSTM",
}


def generate_worst_failures():
    atb_surprisal_df_ids = [
        f"ATB__model__{model}__cfg_tiny_{_GRAMMAR_VERSION}" for model in _MODELS
    ]
    pg_surprisal_df_ids = [
        f"PG__model__{model}__cfg_small_{_GRAMMAR_VERSION}" for model in _MODELS
    ]

    analysis.worst_failures_to_latex_multiple(
        surprisal_df_ids=atb_surprisal_df_ids,
        n=5,
        unique_only=True,
        filename="appendix-failures-atb.tex",
        model_labels=_MODEL_LABELS,
    )

    analysis.worst_failures_to_latex_multiple(
        surprisal_df_ids=pg_surprisal_df_ids,
        n=5,
        unique_only=True,
        filename="appendix-failures-pg.tex",
        model_labels=_MODEL_LABELS,
    )


if __name__ == "__main__":
    generate_worst_failures()
