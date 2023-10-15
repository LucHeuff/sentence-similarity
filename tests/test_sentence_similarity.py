import itertools
import string

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite
from pytest import raises

from sentence_similarity.translator import (
    TokenizeFunction,
    create_default_translator,
    tokenize_characters,
    tokenize_on_spaces,
)

tokenizers = [tokenize_on_spaces, tokenize_characters]
ALPHABET = string.ascii_letters + string.digits + string.punctuation.replace("\\", "")

WORD_LENGTH = 20
VOCAB_LENGTH = 15
SENTENCE_LENGTH = 10

# Nomenclature:
# - a generator is a sampler that is meant to be reused in multiple places,
#   but not directly as a strategy for a test.
# - a strategy is a sampler that is bespoke to a function test.
#   Strategies can output multiple objects.

# ---- standard generators -----


@composite
def word_generator(draw) -> str:
    """Generates a single 'word' that is not allowed to be only empty spaces"""
    word = draw(st.text(ALPHABET, min_size=1, max_size=WORD_LENGTH))
    # assume(word != " ") # words are not allowed to be spaces
    return word


@composite
def sentence_generator(draw) -> str:
    """Generates a sentence of one or more 'words"""
    words = draw(st.lists(word_generator(), min_size=1))
    sentence = " ".join(words)
    # assume(sentence != "") # not allowed to be empty, I'm just assuming sentences are never empty
    return sentence


# ---- Testing numericalization ----
from sentence_similarity.sentence_similarity import _numericalize


@composite
def sentences_generator(
    draw, min_length: int = 1, max_length: int = SENTENCE_LENGTH
) -> list[str]:
    """Generate a list of unique sentences"""
    return draw(
        st.lists(
            sentence_generator(), min_size=min_length, max_size=max_length, unique=True
        )
    )


@given(sentences=sentences_generator(), method=st.sampled_from(tokenizers))
def test_numericalize(sentences, method):
    """
    - Test whether sentences are converted to numpy.ndarrays
    - Test if the same number of sentences are numericalised as are entered in the numericalisation
    """
    translator = create_default_translator(sentences, method)
    numericalized = _numericalize(sentences, translator)
    assert all(isinstance(sentence, np.ndarray) for sentence in numericalized)
    assert len(numericalized) == len(sentences)


# ---- Testing one-hot encoding ----
from sentence_similarity.sentence_similarity import _one_hot_sentence


@composite
def numbered_sentence_generator(
    draw,
    min_length: int = 1,
    max_length: int = SENTENCE_LENGTH,
    vocab_length: int = VOCAB_LENGTH,
) -> np.ndarray:
    """Generates a 'sentence' as a list of numbers as a np.ndarray"""
    nums = st.integers(min_value=0, max_value=vocab_length)
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    return draw(arrays(dtype=np.int64, shape=length, elements=nums))


@given(num_sentence=numbered_sentence_generator())
def test_one_hot_sentence(num_sentence: np.ndarray):
    """
    - Test if the one-hot-encoded matrix has dimensions of (vocab_length, max_sentence_length)
    - Test if the correct index in each column is one-hot-encoded
    """
    vocab_length = (
        num_sentence.max() + 1
    )  # making sure that if the sentence is [0] this is interpreted as a length of 1
    max_sentence_length = num_sentence.size

    encoded = _one_hot_sentence(num_sentence, vocab_length, max_sentence_length)

    assert encoded.shape == (vocab_length, max_sentence_length)
    assert np.array_equal(num_sentence, encoded.argmax(axis=0))


# ---- Testing weight matrix -----
from sentence_similarity.sentence_similarity import _weight_matrix


@given(
    size=st.integers(min_value=2, max_value=10),
    min=st.floats(min_value=0, max_value=0.99),
)
def test_weight_matrix(size: int, min: float):
    """
    - Test if the weight matrix has the desired size
    - Test if the highest value in the matrix is 1
    - Test if the smallest value has the desired value
    - Test if the matrix is symmetric along the diagonal
    - Test if the diagonal is all ones
    """
    weight_matrix = _weight_matrix(size, min)

    assert weight_matrix.shape == (size, size)
    assert weight_matrix.max() == 1.0
    assert weight_matrix.min() == min
    assert np.array_equal(weight_matrix.T, weight_matrix)
    assert all(np.diag(weight_matrix))


@given(
    size=st.integers(min_value=2, max_value=10),
    min=st.floats().filter(
        lambda x: not 0.0 <= x <= 1.0
    ),  # generating floats outside of the range [0, 1]
)
def test_weight_matrix_exception(size: int, min: float):
    """
    - Test if the weight matrix throws an exception if the min value is outside the range [0, 1]
    """
    with raises(ValueError):
        _weight_matrix(size, min)


# ---- Testing einsum ----
from sentence_similarity.sentence_similarity import _einsum


@composite
def numbered_sentences_generator(
    draw, sentence_length: int = SENTENCE_LENGTH, vocab_length: int = VOCAB_LENGTH
) -> list[np.ndarray]:
    """Generate a list of 'numbered_sentence's"""
    size = sentence_length
    return draw(
        st.lists(
            numbered_sentence_generator(
                min_length=size, max_length=size, vocab_length=vocab_length
            ),
            min_size=2,
        )
    )


@composite
def einsum_strategy(draw) -> tuple[list[np.ndarray], int, int]:
    """Generates a list of numbered sentences,
    a vocab length and a sentence length for testing einsum calculations"""
    vocab_length = draw(st.integers(min_value=1, max_value=30))
    sentence_length = draw(st.integers(min_value=2, max_value=30))
    sentences = draw(
        numbered_sentences_generator(
            sentence_length=sentence_length, vocab_length=vocab_length
        )
    )
    assume(
        sentences != []
    )  # Assuming sentences are not empty because they are annoying
    return sentences, vocab_length + 1, sentence_length


# somewhat more of an integeration test but generating the right tensors is really annoying
@given(einsum_strategy())
def test_einsum(data: tuple[list[np.ndarray], int, int]):
    """Pseudo-integration test of einsum
    - Test whether the einsum result has the correct shape
    - Test whether the diagonal has all ones
    - Test whether the minimum value is not smaller than 0
    """
    sentences, vocab_length, sentence_length = data

    one_hot_sentences = [
        _one_hot_sentence(sentence, vocab_length, sentence_length)
        for sentence in sentences
    ]
    tensor = np.stack(one_hot_sentences)
    weight_matrix = _weight_matrix(size=sentence_length, minimum=0.2)

    einsum = _einsum(tensor, weight_matrix)

    n = len(sentences)
    assert einsum.shape == (n, n)
    assert all(np.diag(einsum))
    assert einsum.min() >= 0


# ---- Testing to_dataframe ----
from sentence_similarity.sentence_similarity import _to_dataframe


@composite
def dataframe_strategy(draw) -> tuple[list[str], np.ndarray]:
    """Generates fake einsum output that can be converted to a dataframe.
    Allows testing whether the right values also end up in the right places
    """
    sents = draw(sentences_generator(min_length=2))
    # Generating a fake einsum matrix
    data = draw(
        arrays(
            dtype=np.float64,
            shape=(len(sents), len(sents)),
            elements=st.floats(min_value=0, allow_infinity=False),
            unique=True,
        )
    )
    return sents, data


@given(dataframe_strategy())
def test_to_dataframe(data: tuple[list[str], np.ndarray]):
    """
    - Test whether the dataframe has the correct shape
    - Test whether the correct columns are present
    - Test whether all the sentences are present in both [sentence] and [other_sentence] columns
    - Test whether similarity scores end up in the correct place
    """
    sentences, similarity = data

    df = _to_dataframe(sentences, similarity)

    assert df.shape == (len(sentences) ** 2, 3)
    assert np.array_equal(df.columns, ["sentence", "other_sentence", "similarity"])
    assert np.array_equal(df.sentence.unique(), sentences)
    assert np.array_equal(df.other_sentence.unique(), sentences)

    # checking whether values have ended up in the right place
    indices = list(range(len(sentences)))
    for i, j in itertools.product(indices, indices):
        sentence = sentences[i]
        other_sentence = sentences[j]
        row = df.query("sentence == @sentence & other_sentence == @other_sentence")
        assert row.similarity.values == similarity[i, j]


# ---- integration test of sentence_similarity ----
from sentence_similarity.sentence_similarity import sentence_similarity


@given(
    sentences=sentences_generator(),
    tokenizer=st.sampled_from(tokenizers),
    weight_matrix_min=st.floats(min_value=0, max_value=0.999),
)
def test_sentence_similarity(
    sentences: list[str], tokenizer: TokenizeFunction, weight_matrix_min: float
):
    """Integration test of sentence similarity:
    # - Test whether similarity data frame has the correct shape
    # - Test whether each sentence is in both [sentence] and [other_sentence] columns
    # - Test whether the provided sentences all show up in the [sentence] column
    # - Test whether comparing each sentence with itself results in a similarity score of 1
    """
    similarity = sentence_similarity(
        sentences, tokenizer, weight_matrix_min=weight_matrix_min
    )

    assert similarity.shape == (len(sentences) ** 2, 3)
    assert np.array_equal(
        similarity.sentence.unique(), similarity.other_sentence.unique()
    )
    assert np.array_equal(similarity.sentence.unique(), sentences)
    assert similarity.query("sentence == other_sentence").similarity.all()

