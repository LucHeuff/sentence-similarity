"""Contains tests for translator.py."""

import string
from dataclasses import dataclass
from itertools import chain
from typing import Callable

import hypothesis.strategies as st
import pytest
from hypothesis import given
from hypothesis.strategies import DrawFn, composite
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

PUNCTUATION = r"%&'*+,-./:;=?@^`|~"
ALPHABET = string.ascii_letters + string.digits
WHITESPACE = string.whitespace

# Nomenclature:
# - a generator is a sampler that is meant to be reused in multiple places,
#   but not directly as a strategy for a test.
# - a strategy is a sampler that is bespoke to a function test.
#   Strategies can output multiple objects.

Sampler = Callable[[], st.SearchStrategy[str]]


@composite
def character_generator(draw: DrawFn) -> str:
    """Generate a single character, including WHITESPACE."""
    return draw(
        st.text(ALPHABET + WHITESPACE + PUNCTUATION, min_size=1, max_size=1)
    )


@composite
def word_generator(draw: DrawFn, min_size: int = 1) -> str:
    """Generate a 'word' consisting of multiple characters, excluding WHITESPACE."""
    return draw(st.text(ALPHABET, min_size=min_size, max_size=10))


@composite
def punctuation_generator(draw: DrawFn) -> str:
    """Generate a single character of punctuation."""
    return draw(st.text(PUNCTUATION, min_size=1, max_size=1))


# ---- Testing tokenizer functions ----
from sentence_similarity.translator import (
    StringDistanceVocabError,
    TokenizeFunction,
    Translator,
    Vocab,
    VocabUniqueTokensError,
    create_default_vocab,
    create_string_distance_vocab,
    create_synonym_vocab,
    tokenize_characters,
    tokenize_on_spaces,
    tokenize_words,
)


@composite
def tokenize_on_spaces_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of words and combines those words into a sentence with a WHITESPACE.

    Allows testing whether tokenization correctly recovers the words that went into the sentence.
    """
    words = draw(st.lists(word_generator(), min_size=1, unique=True))
    sentence = " ".join(words)
    return words, sentence


@given(data=tokenize_on_spaces_strategy())
def test_tokenize_on_spaces(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on spaces returns the words that went into the sentence."""
    words, sentence = data
    assert tokenize_on_spaces(sentence) == words


@composite
def tokenize_character_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of characters and combines these characters into a sentence by pasting them together directly.

    Allows testing whether tokenization correctly recovers the characters that went into the sentence.
    """
    characters = draw(st.lists(character_generator(), min_size=1, unique=True))
    sentence = "".join(characters)
    return characters, sentence


@given(data=tokenize_character_strategy())
def test_tokenize_characters(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on characters returns the charactes that went into the sentence."""
    characters, sentence = data
    # testing whether the tokenizer returns each character
    assert tokenize_characters(sentence) == characters


@composite
def tokenize_words_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of words, and puts random punctuation behind each word.

    Allows testing whether tokenize_words correctly splits off punctuation
    """
    words = draw(st.lists(word_generator(), min_size=1, unique=True))
    # putting punctuation after each word, easiest test
    punctuation = draw(
        st.lists(
            punctuation_generator(),
            min_size=len(words),
            max_size=len(words),
            unique=True,
        )
    )
    # making the list of tokens as I expect them to be returned
    tokens = list(chain(*zip(words, punctuation)))
    # making a list of tokens where the punctuation is pasted right behind the word
    tokens_into_sentence = [
        "".join([word, punct]) for (word, punct) in zip(words, punctuation)
    ]
    sentence = " ".join(tokens_into_sentence)
    return tokens, sentence


@given(data=tokenize_words_strategy())
def test_tokenize_words(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on words returns the words and punctuation that went into the sentence."""
    tokens, sentence = data
    assert tokenize_words(sentence) == tokens


# ---- Testing Vocab


@dataclass
class VocabComponents:
    """Contains components for testing Vocab."""

    vocab: dict[str, int]
    vocab_length: int
    fail_tokens: bool
    fail_min: bool
    fail_consecutive: bool


@composite
def vocab_strategy(draw: DrawFn) -> VocabComponents:
    """Generate a vocab and failure cases for Vocab generation.

    Args:
    ----
        draw: hypothesis draw function

    Returns:
    -------
        VocabComponents for test

    """
    fail_min = draw(st.booleans())
    fail_consecutive = draw(st.booleans())
    fail_tokens = draw(st.booleans())
    size = draw(st.integers(min_value=1 + fail_consecutive, max_value=5)) * (
        not fail_tokens
    )
    words = draw(
        st.lists(word_generator(), min_size=size, max_size=size, unique=True)
    )
    vocab = {
        token: (i - (1 * fail_min)) * (1 + fail_consecutive)
        for (i, token) in enumerate(words)
    }
    return VocabComponents(vocab, size, fail_tokens, fail_min, fail_consecutive)


@given(comp=vocab_strategy())
def test_vocab(comp: VocabComponents) -> None:
    """Test whether Vocab creation works as intended."""
    if comp.fail_min or comp.fail_consecutive or comp.fail_tokens:
        with pytest.raises(AssertionError):
            vocab = Vocab(comp.vocab)
    else:
        vocab = Vocab(comp.vocab)
        assert len(vocab) == comp.vocab_length
        for token in comp.vocab:
            assert token in vocab
            assert vocab[token] == comp.vocab[token]
        # checking if __contains__ works as intended.
        # concatenating all tokens and adding a character since that should
        # never randomly appear in the vocab.
        assert not "".join(comp.vocab.keys()) + "a" in vocab


TOKENIZERS = [tokenize_words, tokenize_characters, tokenize_on_spaces]

TOKENIZER_STRATEGY = {
    tokenize_words: tokenize_words_strategy,
    tokenize_on_spaces: tokenize_on_spaces_strategy,
    tokenize_characters: tokenize_character_strategy,
}


def test_create_default_vocab() -> None:
    """Test whether create_default_vocab given correct inputs generates."""
    sentences = [
        "Dit is een handmatige test",
        "Dit is nodig omdat hypothesis stom doet",
        "Dus dan maar een handmatige set zinnen",
    ]
    tokens = ["Dit", "is", "een", "handmatige"]
    not_tokens = [
        "test",
        "nodig",
        "omdat",
        "hypothesis",
        "stom",
        "doet",
        "Dus",
        "dan",
        "maar",
        "set",
        "zinnen",
    ]

    vocab = create_default_vocab(sentences, tokenize_words)
    for token in tokens:
        assert token in vocab
    for token in not_tokens:
        assert token not in vocab


def test_create_default_vocab_exception() -> None:
    """Test whether create_default_vocab raises an error if all tokens appear only once."""
    sentences = ["alle woorden komen", "maar een keer voor"]
    with pytest.raises(VocabUniqueTokensError):
        create_default_vocab(sentences, tokenize_words)


def test_create_synonym_vocab() -> None:
    """Test whether create_synonym_vocab() given correct inputs generates."""
    sentences = [
        "Dit is een handmatige test",
        "Dit is nodig omdat hypothesis stom doet",
        "Dus dan maar een handmatige set zinnen",
    ]
    synonyms = [("Dus", "Dit"), ("test", "set"), ("hypothesis", "stom")]
    tokens = [
        "Dit",
        "Dus",
        "is",
        "een",
        "handmatige",
        "test",
        "set",
        "hypothesis",
        "stom",
    ]
    not_tokens = [
        "nodig",
        "omdat",
        "doet",
        "dan",
        "maar",
        "zinnen",
    ]
    vocab = create_synonym_vocab(sentences, synonyms, tokenize_words)
    for token in tokens:
        assert token in vocab
    for token in not_tokens:
        assert token not in vocab


def test_string_distance_vocab() -> None:
    """Test whether string distance vocab generates like it should."""
    sentences = ["Dit is lastig", "Dus as anders", "Den vis boven"]
    tokens = ["Dit", "Dus", "Den", "is", "as"]
    not_tokens = ["lastig", "anders", "boven"]
    vocab = create_string_distance_vocab(sentences, 3)
    for token in tokens:
        assert token in vocab
    for token in not_tokens:
        assert token not in vocab


def test_string_distance_vocab_exception() -> None:
    """Test whether string_distance_vocab() raises the correct exception when a non-metric string distance object is provided."""
    sentences = ["Dit is lastig", "Dus as anders", "Den vis boven"]
    with pytest.raises(StringDistanceVocabError):
        create_string_distance_vocab(
            sentences,
            2,
            distance_function=NormalizedLevenshtein(),  # type: ignore
        )


# ---- Test Translator


@dataclass
class TranslatorComponents:
    """Components for testing Translator."""

    tokenizer: TokenizeFunction
    vocab: Vocab
    vocab_length: int
    sentence: str
    numbers: list[int]


@composite
def translator_strategy(draw: DrawFn) -> TranslatorComponents:
    """Draw from strategy for testing Translator."""
    tokenizer = draw(st.sampled_from(TOKENIZERS))
    tokens, sentence = draw(TOKENIZER_STRATEGY[tokenizer]())
    vocab = {token: i for (i, token) in enumerate(tokens)}
    vocab_length = len(tokens)
    numbers = list(range(vocab_length))
    return TranslatorComponents(
        tokenizer, Vocab(vocab), vocab_length, sentence, numbers
    )


@given(comp=translator_strategy())
def test_translator(comp: TranslatorComponents) -> None:
    """Test whether Translator creation works as intended."""
    translator = Translator(comp.tokenizer, comp.vocab)
    assert len(translator) == comp.vocab_length
    assert translator.encode(comp.sentence) == comp.numbers
