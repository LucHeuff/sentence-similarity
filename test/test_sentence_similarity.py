import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import composite, lists, text, integers, floats, sampled_from, booleans
from pytest import raises
from string import printable

from src.sentence_similarity import _numericalize, _one_hot_sentence, _weight_matrix, _einsum
from src.vocab import Vocab, tokenizer_options

tokenizer_methods = list(tokenizer_options.keys())


# * Testing numericalization

@composite
def sentence(draw) -> str:
    words = draw(lists(text(printable, min_size=1, max_size=10))) # not allowed to be empty, I'm just assuming sentences are never empty
    return " ".join(words)

@composite
def sentences(draw) -> list[str]:
    sentences = draw(lists(sentence()))
    return [sentence for sentence in sentences if not sentence == ''] # really making sure hypothesis does not keep sneakering empty sentences past me


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

# * Testing one-hot encoding

@composite
def numbered_sentence(draw, min_size: int=1, max_size: int=20, vocab_length: int=25) -> np.ndarray:
    nums = draw(lists(integers(min_value=1, max_value=vocab_length), min_size=min_size, max_size=max_size)) # max value should not be too high for flaky tests
    return np.asarray(nums)

@composite
def numbered_sentences(draw, vocab_length: int=25, sentence_length: int=20) -> list[np.ndarray]:
    size = sentence_length
    return draw(lists(numbered_sentence(min_size=size, max_size=size, vocab_length=vocab_length), min_size=2))


@given(
        num_sentence=numbered_sentence(),
       )
def test_one_hot_sentence(num_sentence: np.ndarray):
    vocab_length = num_sentence.max() + 1 # * adding one because vocab will always have <unk> added
    max_sentence_length = num_sentence.size

    encoded = _one_hot_sentence(num_sentence, vocab_length, max_sentence_length)

    assert encoded.shape == (vocab_length, max_sentence_length)
    assert np.array_equal(num_sentence, encoded.argmax(axis=0))

# * Testing weight matrix

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



# * Testing einsum
@composite
def einsum_data(draw) -> tuple[list[np.ndarray], int, int]:
    vocab_length = draw(integers(min_value=1, max_value=30))
    sentence_length = draw(integers(min_value=2, max_value=30))
    sentences = draw(numbered_sentences(vocab_length, sentence_length))
    assume(sentences != [])
    return sentences, vocab_length, sentence_length


# * somewhat more of an integeration test but generating the right tensors is really annoying
@given(einsum_data())
def test_einsum(data: tuple[list[np.ndarray], int, int]):
    sentences, vocab_length, sentence_length = data
    vocab_length += 1 # adding 1 for the unknown token

    one_hot_sentences = [_one_hot_sentence(sentence, vocab_length, sentence_length) for sentence in sentences]
    tensor = np.stack(one_hot_sentences)
    weight_matrix = _weight_matrix(size=sentence_length, min=0.2)

    einsum = _einsum(tensor, weight_matrix)

    n = len(sentences)
    assert einsum.shape == (n, n)
    assert all(np.diag(einsum))