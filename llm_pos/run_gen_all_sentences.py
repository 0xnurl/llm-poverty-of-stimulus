import collections
import itertools

import markdown

import lib


def _sentence_to_markdown(sentence: lib.Sentence) -> str:
    return lib.format_sentence(
        sentence,
        add_grammaticality=False,
        left_region_marker="**",
        right_region_marker="**",
        add_critical_surprisal=False,
    )


def write_sentences_plaintext(
    phenomenon,
    names_size,
    conditions=None,
    retraining_corpus=False,
):
    grammar_version = lib.get_grammar_version()
    filename = f"./data/all_sentences_{phenomenon}_{names_size}_{grammar_version}"
    if retraining_corpus:
        filename += "_retraining"
    with open(f"{filename}.txt", "w") as f:
        for sentence in lib.iterate_phenomenon_sentences(
            phenomenon,
            names_size,
            conditions=conditions,
            retraining_corpus=retraining_corpus,
        ):
            s = lib.format_sentence(
                sentence,
                add_grammaticality=False,
                left_region_marker="",
                right_region_marker="",
                add_critical_surprisal=False,
            )
            f.write(f"{s}\n")


def _write_sentences_html(
    names_size, phenomena=None, retraining_corpus=False, conditions=None
):
    phenomenon_to_grammars = lib.load_grammars(
        names_size, retraining_corpus=retraining_corpus
    )

    phenomenon_to_sentences = {}
    for phenomenon in phenomenon_to_grammars:
        if phenomena is not None:
            if phenomenon not in phenomena:
                continue
        phenomenon_to_sentences[phenomenon] = {}
        for condition, grammars in phenomenon_to_grammars[phenomenon].items():
            if conditions is not None:
                if condition not in conditions:
                    continue
            phenomenon_to_sentences[phenomenon][condition] = []
            for grammar in grammars:
                condition_grammar_sentences = tuple(
                    lib.generate_sentences_from_cfg(grammar)
                )
                phenomenon_to_sentences[phenomenon][condition].append(
                    condition_grammar_sentences
                )

    filename = f"../data/all_sentences_{names_size}_names"
    if retraining_corpus:
        filename += "_retraining"
    with open(f"{filename}.md", "w") as all_sentences_file:
        all_sentences_file.write(
            f"# All sentences {'retraining corpus ' if retraining_corpus else ' '}({names_size} name list)\n"
        )

        for p, phenomenon in enumerate(phenomenon_to_sentences):
            total_phenomenon_sentences = sum(
                map(
                    len,
                    itertools.chain(*phenomenon_to_sentences[phenomenon].values()),
                )
            )

            all_sentences_file.write(
                f"## {p + 1}. {phenomenon} ({total_phenomenon_sentences:,})\n"
            )

            for c, (condition_name, condition_grammar_sentences) in enumerate(
                phenomenon_to_sentences[phenomenon].items()
            ):
                total_condition_sentences = sum(map(len, condition_grammar_sentences))
                all_sentences_file.write(
                    f"\n### {condition_name} ({total_condition_sentences:,})\n"
                )

                for g, sentences in enumerate(condition_grammar_sentences):
                    all_sentences_file.write(
                        f"\n#### Grammar {g + 1} ({len(sentences):,})\n\n"
                    )

                    for i, sentence in enumerate(sentences):
                        s = _sentence_to_markdown(sentence)
                        all_sentences_file.write(
                            f"- {p+1}.{chr(97 + c)}.{g+1}.{i+1}. {s}\n"
                        )

    with open(f"{filename}.md", "r") as f:
        html = markdown.markdown("[TOC]\n\n" + f.read(), extensions=["toc"])

    with open(f"{filename}.html", "w") as f:
        f.write(html)


def _cleanup(s):
    return s.replace("_", r"\_")


def _grammar_to_dict(g):
    d = collections.defaultdict(list)
    for production in g._productions:
        left = _cleanup(str(production._lhs))
        if left == lib.CRITICAL_REGION_DERIVATION_SYMBOL:
            continue
        rights = list(
            map(
                _cleanup,
                filter(
                    lambda x: x not in {lib.CRITICAL_REGION_DERIVATION_SYMBOL},
                    map(str, production._rhs),
                ),
            )
        )
        if d[left]:
            d[left].append("|")
        d[left] += rights
    return d


def _cfg_to_latex(phenomenon, names_size, retraining_corpus=False):
    # \newcommand{\angles}[1]{\langle {#1} \rangle\ }

    grammars = lib.load_grammars(names_size, retraining_corpus=retraining_corpus)
    phenomenon_grammars = grammars[phenomenon]

    merged_grammar_dicts = collections.defaultdict(dict)

    for condition in phenomenon_grammars:
        for g, grammar in enumerate(phenomenon_grammars[condition]):
            grammar_dict = _grammar_to_dict(grammar)
            for l, r in grammar_dict.items():
                if l not in merged_grammar_dicts[g]:
                    merged_grammar_dicts[g][l] = r

    grammars_to_text = {}

    nodes_to_ignore = {"GEN\\_FAKE", "UNGRAMMATICAL"}

    for g, grammar_dict in merged_grammar_dicts.items():
        grammar_text = ""
        for left, rights in grammar_dict.items():
            if left in nodes_to_ignore:
                continue

            row = r"\angles{" + left + r"} \rightarrow "

            right_wrapped = []
            for right in rights:
                if right.upper() == right and right not in {"I", r"\_", "|", "*"}:
                    right_wrapped.append(r"\angles{" + right + r"}")
                elif right != "|":
                    right_wrapped.append(r"\text{`" + right + "'}")
                else:
                    right_wrapped.append(r"|")

            row += r"\ ".join(right_wrapped)

            grammar_text += f"${row}$" + "\n" + r" \\ " + "\n"

        grammars_to_text[g] = grammar_text

    filename = f"appendix-cfgs-{phenomenon.lower()}"
    if retraining_corpus:
        filename += "-retraining"
    with open(f"./data/{filename}.tex", "w") as f:
        header = r"\subsection{" + phenomenon
        if retraining_corpus:
            header += " -- retraining corpus"
        header += "}\n\n"
        f.write(header)

        for g, text in grammars_to_text.items():
            f.write(r"\subsubsection{" + "Grammar " + str(g + 1) + "}\n\n")
            f.write(text)


if __name__ == "__main__":
    _write_sentences_html(
        names_size="tiny",
        phenomena={"ATB"},
        retraining_corpus=False,
    )
    _write_sentences_html(
        names_size="tiny",
        phenomena={"ATB"},
        retraining_corpus=True,
    )
    _write_sentences_html(
        names_size="small",
        phenomena={"PG"},
        retraining_corpus=False,
    )
    _write_sentences_html(
        names_size="small",
        phenomena={"PG"},
        retraining_corpus=True,
    )
