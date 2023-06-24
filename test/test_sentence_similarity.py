import itertools
import string
import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import composite, lists, text, integers, floats, sampled_from, booleans
from hypothesis.extra.numpy import arrays
from pytest import raises

from src.sentence_similarity import _numericalize, _one_hot_sentence, _weight_matrix, _einsum, _to_dataframe, sentence_similarity
from src.translator import create_default_translator, ALLOWED_TOKENIZE_METHODS

WORD_LENGTH = 20
VOCAB_LENGTH = 15
SENTENCE_LENGTH = 10

# TODO fix imports from hypothesis.strategies to st.

# TODO rewrite tests for updates Translator and sentence_similarity()

# * General strategies

alphabet = string.ascii_letters + string.digits + string.punctuation.replace("\\", "")

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

# * Testing numericalization
#  - Test whether sentences are converted to numpy.ndarrays
#  - Test if the same number of sentences are numericalised as are entered in the numericalisation

@composite
def sentences(draw, min_length: int=1, max_length: int=SENTENCE_LENGTH) -> list[str]:
    return draw(lists(sentence(), min_size=min_length, max_size=max_length, unique=True))

@given(sentences=sentences(), method=sampled_from(ALLOWED_TOKENIZE_METHODS))
def test_numericalize(sentences, method):
    translator = create_default_translator(sentences, method)
    numericalized = _numericalize(sentences, translator)
    assert all(isinstance(sentence, np.ndarray) for sentence in numericalized)
    assert len(numericalized) == len(sentences)

# TODO fix crap with numbered sentence and vocab lengths not lining up

# * Testing one-hot encoding
# - Test if the one-hot-encoded matrix has dimensions of (vocab_length, max_sentence_length)
# - Test if the correct index in each column is one-hot-encoded

# @composite
# def numbered_sentence(draw, min_length: int=1, max_length: int=SENTENCE_LENGTH, vocab_length: int=VOCAB_LENGTH) -> np.ndarray:
#     nums = draw(lists(integers(min_value=0, max_value=vocab_length), min_size=min_length, max_size=max_length)) # max value should not be too high for flaky tests
#     return np.asarray(nums)

@composite
def numbered_sentence(draw, min_length: int=1, max_length: int=SENTENCE_LENGTH, vocab_length: int=VOCAB_LENGTH) -> np.ndarray:
    nums = integers(min_value=0, max_value=vocab_length)
    length = draw(integers(min_value=min_length, max_value=max_length))
    return draw(arrays(dtype=np.int64, shape=length, elements=nums))

@given(num_sentence=numbered_sentence())
def test_one_hot_sentence(num_sentence: np.ndarray):
    vocab_length = num_sentence.max() + 1  # making sure that if the sentence is [0] this is interpreted as a length of 1
    max_sentence_length = num_sentence.size

    encoded = _one_hot_sentence(num_sentence, vocab_length, max_sentence_length)

    assert encoded.shape == (vocab_length, max_sentence_length)
    assert np.array_equal(num_sentence, encoded.argmax(axis=0))

# * Testing weight matrix
# - Test if the weight matrix has the desired size
# - Test if the highest value in the matrix is 1
# - Test if the smallest value has the desired value
# - Test if the matrix is symmetric along the diagonal
# - Test if the diagonal is all ones

# - Test if the weight matrix throws an exception if the min value is outside the range [0, 1]

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
    assert all(np.diag(weight_matrix))

@given(
    size=integers(min_value=2, max_value=10),
    min=floats().filter(lambda x: not 0. <= x <= 1.)  # generating floats outside of the range [0, 1]
)
def test_weight_matrix_exception(size: int, min: float):
    with raises(ValueError):
        _weight_matrix(size, min)


# * Testing einsum
# - Test whether the einsum result has the correct shape
# - Test whether the diagonal has all ones
# - Test whether the minimum value is not smaller than 0

@composite
def numbered_sentences(draw, sentence_length: int=SENTENCE_LENGTH, vocab_length: int=VOCAB_LENGTH) -> list[np.ndarray]:
    size = sentence_length
    return draw(lists(numbered_sentence(min_length=size, max_length=size, vocab_length=vocab_length), min_size=2))

@composite
def einsum_data(draw) -> tuple[list[np.ndarray], int, int]:
    vocab_length = draw(integers(min_value=1, max_value=30))
    sentence_length = draw(integers(min_value=2, max_value=30))
    sentences = draw(numbered_sentences(sentence_length=sentence_length, vocab_length=vocab_length))
    assume(sentences != []) # Assuming sentences are not empty because they are annoying
    return sentences, vocab_length + 1, sentence_length

# somewhat more of an integeration test but generating the right tensors is really annoying
@given(einsum_data())
def test_einsum(data: tuple[list[np.ndarray], int, int]):
    sentences, vocab_length, sentence_length = data

    one_hot_sentences = [_one_hot_sentence(sentence, vocab_length, sentence_length) for sentence in sentences]
    tensor = np.stack(one_hot_sentences)
    weight_matrix = _weight_matrix(size=sentence_length, min=0.2)

    einsum = _einsum(tensor, weight_matrix)

    n = len(sentences)
    assert einsum.shape == (n, n)
    assert all(np.diag(einsum))
    assert einsum.min() >= 0


# * Testing to_dataframe
# - Test whether the dataframe has the correct shape
# - Test whether the correct columns are present
# - Test whether all the sentences are present in both [sentence] and [other_sentence] columns
# - Test whether similarity scores end up in the correct place

@composite
def dataframe_data(draw) -> tuple[list[str], np.ndarray]:
    sents = draw(sentences(min_length=2))
    # Generating a fake einsum matrix
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
# - Test whether similarity data frame has the correct shape
# - Test whether each sentence is in both [sentence] and [other_sentence] columns
# - Test whether the provided sentences all show up in the [sentence] column
# - Test whether comparing each sentence with itself results in a similarity score of 1

@given(
    sentences=sentences(),
    tokenize_method=sampled_from(ALLOWED_TOKENIZE_METHODS),
    weight_matrix_min=floats(min_value=0, max_value=0.999)
)
def test_sentence_similarity(sentences: list[str], tokenize_method: str, weight_matrix_min: float):
    similarity = sentence_similarity(sentences, tokenize_method, weight_matrix_min=weight_matrix_min)

    assert similarity.shape == (len(sentences)**2, 3)
    assert np.array_equal(similarity.sentence.unique(), similarity.other_sentence.unique())
    assert np.array_equal(similarity.sentence.unique(), sentences)
    assert similarity.query('sentence == other_sentence').similarity.all()
