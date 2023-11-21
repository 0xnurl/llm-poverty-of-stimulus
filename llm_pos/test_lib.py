import random
import unittest

import nltk
import pytest
from nltk.parse import generate

import experiments
import lib


class TestLib(unittest.TestCase):
    def test_grammar_to_sentence(self):
        cfg = """
        S -> UNGRAMMATICAL NAME POS V "_" ADJUNCT "_"
        UNGRAMMATICAL -> "*"
        NAME -> "Mary"
        POS -> "'s" 
        V -> "talking to bill"
        ADJUNCT -> "yesterday"
        """
        cfg_grammar = nltk.CFG.fromstring(cfg)
        sentences = tuple(lib.generate_sentences_from_cfg(cfg_grammar))
        s = sentences[0]
        assert str(s) == "Mary's talking to bill yesterday"
        assert not s.grammatical
        assert tuple(map(lambda x: x.critical_region, s.preprocessed_tokens)) == (
            False,
            False,
            False,
            False,
            True,
        )

    def test_possessive_tokenization(self):
        cfg = """
        S -> NAME POS
        NAME -> "Mary"
        POS -> "'s" 
        """
        cfg_grammar = nltk.CFG.fromstring(cfg)
        cfg_output = tuple(generate.generate(cfg_grammar))[0]
        sentence = lib.grammar_output_to_sentence(cfg_output)
        assert str(sentence) == "Mary's"
        assert len(sentence.original_token_strings) == 1
        assert len(sentence.preprocessed_tokens) == 1

        cfg = """
        S -> NAME POS F
        NAME -> "Mary"
        POS -> "'s" 
        F -> "basketball"
        """
        cfg_grammar = nltk.CFG.fromstring(cfg)
        cfg_output = tuple(generate.generate(cfg_grammar))[0]
        sentence = lib.grammar_output_to_sentence(cfg_output)
        assert str(sentence) == "Mary's basketball"
        assert len(sentence.original_token_strings) == 2
        assert len(sentence.preprocessed_tokens) == 2

    def test_multiword_derivation(self):
        cfg = """
            S -> S1 S2
            S1 -> "What do you"
            S2 -> "think"
            """
        cfg_grammar = nltk.CFG.fromstring(cfg)
        cfg_output = tuple(generate.generate(cfg_grammar))[0]
        sentence = lib.grammar_output_to_sentence(cfg_output)
        assert str(sentence) == "What do you think", sentence
        assert len(sentence.preprocessed_tokens) == 4
        assert len(sentence.original_token_strings) == 4
