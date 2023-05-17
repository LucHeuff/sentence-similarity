import hypothesis
import string
import numpy as np
from hypothesis import given
from hypothesis.strategies import SearchStrategy, composite, lists, text, integers, floats, sampled_from, booleans
from pytest import approx, raises
from typing import Callable

from src.sentence_similarity import _numericalize, _one_hot_sentence, _weight_matrix, _einsum
from src.vocab import Vocab, tokenizer_options

# hypothesis.settings(deadline=1000) # attempt to avoid flaky tests

tokenizer_methods = list(tokenizer_options.keys())
alphabet = string.ascii_letters + string.digits + string.punctuation # allowing everything but spaces since those are a token splitting condition

@composite
def sentence(draw) -> str:
    words = draw(lists(text(alphabet, min_size=1, max_size=10))) # not allowed to be empty, I'm just assuming sentences are never empty
    return " ".join(words)

@composite
def numbered_sentence(draw, min_size: int=1, max_size: int=20, vocab_length: int=25) -> np.ndarray:
    nums = draw(lists(integers(min_value=1, max_value=vocab_length), min_size=min_size, max_size=max_size)) # max value should not be too high for flaky tests
    return np.asarray(nums)

@composite
def sentences(draw) -> list[str]:
    sentences = draw(lists(sentence()))
    return [sentence for sentence in sentences if not sentence == ''] # really making sure hypothesis does not keep sneakering empty sentences past me

@composite
def numbered_sentences(draw, vocab_length: int=25, sentence_length: int=20) -> list[np.ndarray]:
    size = sentence_length
    return draw(lists(numbered_sentence(min_size=size, max_size=size, vocab_length=vocab_length), min_size=3))

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
       )
def test_one_hot_sentence(num_sentence: np.ndarray):
    vocab_length = num_sentence.max() + 1 # * adding one because vocab will always have <unk> added
    max_sentence_length = num_sentence.size

    encoded = _one_hot_sentence(num_sentence, vocab_length, max_sentence_length)

    assert encoded.shape == (vocab_length, max_sentence_length)
    assert np.array_equal(num_sentence, encoded.argmax(axis=0))


@given(
    size=integers(min_value=2, max_value=10),
    min=floats(min_value=0, max_value=0.9990234375, width=16) #* operating on 16 bit floats
)
def test_weight_matrix(size: int, min: float):
    weight_matrix = _weight_matrix(size, min)

    assert weight_matrix.shape == (size, size)
    assert weight_matrix.max() == 1.
    assert weight_matrix.min() == min
    assert np.array_equal(weight_matrix.T, weight_matrix)


@given(
    size=integers(min_value=2, max_value=10),
    min=floats(min_value=1)
)
def test_weight_matrix_exception(size: int, min: float):
    with raises(ValueError):
        _weight_matrix(size, min)


# * somewhat more of an integeration test but generating the right tensors is really annoying
@given(
        sentences=numbered_sentences(vocab_length=25, sentence_length=20)
        )
def test_einsum(sentences: list[np.ndarray]):
    vocab_length = 25 + 1 # adding 1 for the unknown token
    sentence_length = 20

    one_hot_sentences = [_one_hot_sentence(sentence, vocab_length, sentence_length) for sentence in sentences]
    tensor = np.stack(one_hot_sentences)
    weight_matrix = _weight_matrix(size=20, min=0.2)

    einsum = _einsum(tensor, weight_matrix)

    n = len(sentences)
    assert einsum.shape == (n, n)
    assert all(np.diag(einsum))