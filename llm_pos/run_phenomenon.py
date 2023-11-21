import argparse

import experiments

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-p",
    "--phenomenon",
    dest="phenomenon",
    required=True,
    help=f"Phenomenon name: 'PG', 'ATB'",
)

arg_parser.add_argument(
    "-m",
    "--models",
    dest="models",
    required=True,
    help=f"Model names, comma separated, e.g.: `gpt2, grnn`",
)

arg_parser.add_argument(
    "--names",
    dest="names_size",
    required=True,
    help=f"Names size: 'tiny', 'small', 'large'",
)
arg_parser.add_argument(
    "-g",
    "--grammar",
    dest="grammar_idx",
    required=False,
    default=None,
    type=str,
    help=f"Run specific grammar, zero-indexed, comma-separated. Default: run all grammars in phenomenon.",
)


arguments = arg_parser.parse_args()

if __name__ == "__main__":
    models = tuple(map(str.strip, arguments.models.split(",")))

    if arguments.grammar_idx is not None:
        grammar_idxs = tuple(map(int, map(str.strip, arguments.grammar_idx.split(","))))
    else:
        grammar_idxs = (None,)

    for grammar_idx in grammar_idxs:
        for model in models:
            experiments.run_phenomenon(
                phenomenon=arguments.phenomenon,
                model=model,
                names_size=arguments.names_size,
                grammar_idx=grammar_idx,
            )
