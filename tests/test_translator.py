"""Contains tests for translator.py."""

import string
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain

import hypothesis.strategies as st
import pytest
from hypothesis import given
from hypothesis.strategies import DrawFn, composite
from sentence_similarity.translator import (
    TokenizeFunction,
    Translator,
    Vocab,
    create_default_vocab,
    create_synonym_vocab,
    tokenize_characters,
    tokenize_on_spaces,
    tokenize_words,
)

from tests.generators import character, punctuation, sentence, word

PUNCTUATION = r"%&'*+,-./:;=?@^`|~"
ALPHABET = string.ascii_letters + string.digits
WHITESPACE = string.whitespace

# Nomenclature:
# - a generator is a sampler that is meant to be reused in multiple places,
#   but not directly as a strategy for a test.
# - a strategy is a sampler that is bespoke to a function test.
#   Strategies can output multiple objects.

Sampler = Callable[[], st.SearchStrategy[str]]


# ---- Testing tokenizer functions ----


@composite
def tokenize_on_spaces_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of words and combines those words into a sentence with a WHITESPACE.

    Allows testing whether tokenization correctly recovers the words that went into the sentence.
    """  # noqa: E501
    words = draw(st.lists(word(), min_size=1, unique=True))
    sentence = " ".join(words)
    return words, sentence


@given(data=tokenize_on_spaces_strategy())
def test_tokenize_on_spaces(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on spaces returns the words that went into the sentence."""  # noqa: E501
    words, sentence = data
    assert tokenize_on_spaces(sentence) == words


@composite
def tokenize_character_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of characters and combines these characters into a sentence by pasting them together directly.

    Allows testing whether tokenization correctly recovers the characters that went into the sentence.
    """  # noqa: E501
    characters = draw(st.lists(character(), min_size=1, unique=True))
    sentence = "".join(characters)
    return characters, sentence


@given(data=tokenize_character_strategy())
def test_tokenize_characters(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on characters returns the charactes that went into the sentence."""  # noqa: E501
    characters, sentence = data
    # testing whether the tokenizer returns each character
    assert tokenize_characters(sentence) == characters


@composite
def tokenize_words_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of words, and puts random punctuation behind each word.

    Allows testing whether tokenize_words correctly splits off punctuation
    """
    words = draw(st.lists(word(), min_size=1, unique=True))
    # putting punctuation after each word, easiest test
    punct = draw(
        st.lists(
            punctuation(),
            min_size=len(words),
            max_size=len(words),
            unique=True,
        )
    )
    # making the list of tokens as I expect them to be returned
    tokens = list(chain(*zip(words, punct, strict=False)))
    # making a list of tokens where the punctuation is pasted right behind the word
    tokens_into_sentence = [
        f"{word}{punct}" for (word, punct) in zip(words, punct, strict=False)
    ]
    sentence = " ".join(tokens_into_sentence)
    return tokens, sentence


@given(data=tokenize_words_strategy())
def test_tokenize_words(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on words returns the words and punctuation that went into the sentence."""  # noqa: E501
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
    words = draw(st.lists(word(), min_size=size, max_size=size, unique=True))
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
            Vocab(comp.vocab)
    else:
        vocab = Vocab(comp.vocab)
        assert len(vocab) == comp.vocab_length
        for token in comp.vocab:
            assert token in vocab
            assert vocab[token] == comp.vocab[token]
        # checking if __contains__ works as intended.
        # concatenating all tokens and adding a character since that should
        # never randomly appear in the vocab.
        assert "".join(comp.vocab.keys()) + "a" not in vocab


TOKENIZERS = [tokenize_words, tokenize_characters, tokenize_on_spaces]

TOKENIZER_STRATEGY = {
    tokenize_words: tokenize_words_strategy,
    tokenize_on_spaces: tokenize_on_spaces_strategy,
    tokenize_characters: tokenize_character_strategy,
}


@dataclass
class VocabStrategy:
    """Strategy for testing Vocabs."""

    sentences: list[str]
    tokens: list[str]


@composite
def default_vocab_strategy(draw: DrawFn) -> VocabStrategy:
    """Strategy for testing create_default_vocab."""
    sentences = draw(st.lists(sentence(), min_size=3, max_size=7))
    tokens = list({word for sentence in sentences for word in sentence.split()})
    return VocabStrategy(sentences, tokens)


@given(s=default_vocab_strategy())
def test_create_default_vocab(s: VocabStrategy) -> None:
    """Test whether create_default_vocab given correct inputs generates."""
    vocab = create_default_vocab(s.sentences, tokenize_words)
    assert all(token in vocab for token in s.tokens)


@dataclass
class SynonymVocabStrategy:
    """Extension on VocabStrategy for synonyms."""

    sentences: list[str]
    tokens: list[str]
    synonyms: list[str]


@composite
def synonym_vocab_strategy(draw: DrawFn) -> SynonymVocabStrategy:
    """Strategy for testing create_synonym_vocab."""
    sentences = draw(st.lists(sentence(5), min_size=3, max_size=7))
    tokens = list({word for sentence in sentences for word in sentence.split()})
    n_synonyms = draw(st.integers(3, 5))
    synonyms = draw(
        st.lists(
            st.sampled_from(tokens),
            min_size=n_synonyms,
            max_size=n_synonyms,
            unique=True,
        )
    )

    return SynonymVocabStrategy(sentences, tokens, synonyms)


@given(s=synonym_vocab_strategy())
def test_create_synonym_vocab(s: SynonymVocabStrategy) -> None:
    """Test whether create_synonym_vocab() given correct inputs generates."""
    vocab = create_synonym_vocab(s.sentences, [tuple(s.synonyms)], tokenize_words)
    assert all(token in vocab for token in s.tokens)
    assert len({vocab[syn] for syn in s.synonyms}) == 1, (
        "Synonyms don't all get the same value"
    )


# ---- Test Translator


@dataclass
class TranslatorStrategy:
    """Components for testing Translator."""

    tokenizer: TokenizeFunction
    vocab: Vocab
    vocab_length: int
    sentence: str
    numbers: list[int]


@composite
def translator_strategy(draw: DrawFn) -> TranslatorStrategy:
    """Draw from strategy for testing Translator."""
    tokenizer = draw(st.sampled_from(TOKENIZERS))
    tokens, sentence = draw(TOKENIZER_STRATEGY[tokenizer]())
    vocab = {token: i for (i, token) in enumerate(tokens)}
    vocab_length = len(tokens)
    numbers = list(range(vocab_length))
    return TranslatorStrategy(
        tokenizer, Vocab(vocab), vocab_length, sentence, numbers
    )


@given(s=translator_strategy())
def test_translator(s: TranslatorStrategy) -> None:
    """Test whether Translator creation works as intended."""
    translator = Translator(s.tokenizer, s.vocab)
    assert len(translator) == s.vocab_length
    assert translator.encode(s.sentence) == s.numbers
    # Testing reprs for coverage
    assert isinstance(repr(translator.vocab), str)
    assert isinstance(repr(translator), str)
