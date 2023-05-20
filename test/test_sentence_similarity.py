import itertools
import string
import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import composite, lists, text, integers, floats, sampled_from, booleans
from hypothesis.extra.numpy import arrays
from pytest import raises

from src.sentence_similarity import _numericalize, _one_hot_sentence, _weight_matrix, _einsum, _to_dataframe, sentence_similarity
from src.vocab import Vocab, tokenizer_options

tokenizer_methods = list(tokenizer_options.keys())

WORD_LENGTH = 20
VOCAB_LENGTH = 15
SENTENCE_LENGTH = 10


# * Testing numericalization
alphabet = string.ascii_letters + string.digits + string.punctuation

@composite
def word(draw) -> str:
    word = draw(text(alphabet, min_size=1, max_size=WORD_LENGTH))
    assume(word != " ") # words are not allowed to be spaces 
    return word

@composite
def sentence(draw) -> str:
    words = draw(lists(word())) 
    sentence = " ".join(words)
    assume(sentence != "") # not allowed to be empty, I'm just assuming sentences are never empty
    return sentence

@composite
def sentences(draw, min_length: int=1, max_length: int=SENTENCE_LENGTH) -> list[str]:
    return draw(lists(sentence(), min_size=min_length, max_size=max_length, unique=True))

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
def numbered_sentence(draw, min_length: int=1, max_length: int=SENTENCE_LENGTH, vocab_length: int=VOCAB_LENGTH) -> np.ndarray:
    nums = draw(lists(integers(min_value=1, max_value=vocab_length), min_size=min_length, max_size=max_length)) # max value should not be too high for flaky tests
    return np.asarray(nums)

@composite
def numbered_sentences(draw, sentence_length: int=SENTENCE_LENGTH, vocab_length: int=VOCAB_LENGTH) -> list[np.ndarray]:
    size = sentence_length
    return draw(lists(numbered_sentence(min_length=size, max_length=size, vocab_length=vocab_length), min_size=2))

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
    min=floats(min_value=0, max_value=0.99) 
)
def test_weight_matrix(size: int, min: float):
    weight_matrix = _weight_matrix(size, min)

    assert weight_matrix.shape == (size, size)
    assert weight_matrix.max() == 1.
    assert weight_matrix.min() == min
    assert np.array_equal(weight_matrix.T, weight_matrix)

@given(
    size=integers(min_value=2, max_value=10),
    min=floats().filter(lambda x: not 0. <= x <= 1.)  # generating floats outside of the range 0, 1
)
def test_weight_matrix_exception(size: int, min: float):
    with raises(ValueError):
        _weight_matrix(size, min)


# * Testing einsum
@composite
def einsum_data(draw) -> tuple[list[np.ndarray], int, int]:
    vocab_length = draw(integers(min_value=1, max_value=30))
    sentence_length = draw(integers(min_value=2, max_value=30))
    sentences = draw(numbered_sentences(sentence_length=sentence_length, vocab_length=vocab_length))
    assume(sentences != []) # Assuming sentences are not empty because they are annoying
    return sentences, vocab_length, sentence_length

# somewhat more of an integeration test but generating the right tensors is really annoying
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


# # * Testing to_dataframe
@composite
def dataframe_data(draw) -> tuple[list[str], np.ndarray]:
    sents = draw(sentences(min_length=2))
    data = draw(arrays(
        dtype=np.float64,
        shape=(len(sents), len(sents)),
        elements=floats(min_value=0, allow_infinity=False),
        unique=True))
    return sents, data

@given(dataframe_data())
def test_to_dataframe(data: tuple[list[str], np.ndarray]):
    sentences, similarity = data

    df = _to_dataframe(sentences, similarity)

    assert df.shape == (len(sentences)**2, 3)
    assert np.array_equal(df.columns, ['sentence', 'other_sentence', 'similarity'])
    assert np.array_equal(df.sentence.unique(), sentences) 
    assert np.array_equal(df.other_sentence.unique(), sentences) 

    # checking whether values have ended up in the right place
    indices = list(range(len(sentences)))
    for i, j in itertools.product(indices, indices):
        sentence = sentences[i]
        other_sentence = sentences[j]
        row = df.query("sentence == @sentence & other_sentence == @other_sentence")
        assert row.similarity.values == similarity[i, j]


# * integration test of sentence_similarity

@given(
    sentences=sentences(),
    tokenize_method=sampled_from(tokenizer_methods),
    lower=booleans(),
    weight_matrix_min=floats(min_value=0, max_value=0.999)
)
def test_sentence_similarity(sentences: list[str], tokenize_method: str, lower: bool, weight_matrix_min: float):
    similarity = sentence_similarity(sentences, tokenize_method, lower, weight_matrix_min)

    assert similarity.shape == (len(sentences)**2, 3)
    assert np.array_equal(similarity.sentence.unique(), similarity.other_sentence.unique())
    assert np.array_equal(similarity.sentence.unique(), sentences)
