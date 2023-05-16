import hypothesis
import string
import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import SearchStrategy, composite, lists, text, integers, sampled_from, booleans
from pytest import approx
from typing import Callable

from src.sentence_similarity import _numericalize, _one_hot_sentence
from src.vocab import Vocab, tokenizer_options

# hypothesis.settings(deadline=1000) # attempt to avoid flaky tests

tokenizer_methods = list(tokenizer_options.keys())
alphabet = string.ascii_letters + string.digits + string.punctuation # allowing everything but spaces since those are a token splitting condition

@composite
def sentence(draw) -> str:
    words = draw(lists(text(alphabet, min_size=1, max_size=10))) # not allowed to be empty, I'm just assuming sentences are never empty
    return " ".join(words)

@composite
def numbered_sentence(draw, min_size: int=1, max_size: int=20) -> np.ndarray:
    nums = draw(lists(integers(min_value=1, max_value=25), min_size=min_size, max_size=max_size)) # max value should not be too high for flaky tests
    return np.asarray(nums)

@composite
def sentences(draw) -> list[str]:
    sentences = draw(lists(sentence()))
    return [sentence for sentence in sentences if not sentence == ''] # really making sure hypothesis does not keep sneakering empty sentences past me

@composite
def numbered_sentences(draw) -> list[np.ndarray]:
    return draw(lists(numbered_sentence(), min_size=1))

# def tokenize_words(sentence: str) -> list[str]:
#     return sentence.split()

# def tokenize_characters(sentence: str) -> list[str]:
#     return list(sentence)


# * Testing separate methods
@given(
        sentences=sentences(),
        method=sampled_from(tokenizer_methods),
        lower=booleans()
        )
def test_numericalize(sentences, method, lower):
    vocab = Vocab(sentences, method, lower)
    numericalized = _numericalize(sentences, vocab)
    assert all(isinstance(sentence, np.ndarray) for sentence in numericalized)
    assert len(numericalized) == len(sentences)

@given(
        num_sentence=numbered_sentence(),
        term_frequency=booleans(),
        scale_by_length=booleans()
       )
def test_one_hot_sentence(num_sentence: np.ndarray, term_frequency: bool, scale_by_length: bool):
    vocab_length = num_sentence.max() + 1 # * adding one because vocab will always have <unk> added
    max_sentence_length = num_sentence.size

    encoded = _one_hot_sentence(num_sentence, vocab_length, max_sentence_length, term_frequency, scale_by_length)

    assert encoded.shape == (vocab_length, max_sentence_length)
    assert np.array_equal(num_sentence, encoded.argmax(axis=0))

    if not term_frequency and not scale_by_length:
        assert encoded.sum() == max_sentence_length
    if term_frequency and not scale_by_length:
        assert encoded.sum() == np.unique(num_sentence).size
    if term_frequency and scale_by_length:
        assert encoded.sum() == approx(np.unique(num_sentence).size / num_sentence.size)
