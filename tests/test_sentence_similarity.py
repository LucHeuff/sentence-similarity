"""Contains tests for sentence_similarity.py."""

import itertools
import string
from dataclasses import dataclass

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import DrawFn, composite
from polars.testing import assert_frame_equal
from sentence_similarity.exceptions import (
    DuplicateSentencesError,
    InvalidWeightBaseError,
)
from sentence_similarity.sentence_similarity import (
    _einsum,
    _numericalize,
    _one_hot_sentence,
    _to_dataframe,
    _weight_matrix,
    sentence_similarity,
)
from sentence_similarity.translator import (
    TokenizeFunction,
    Translator,
    create_default_vocab,
    tokenize_characters,
    tokenize_on_spaces,
)

from tests.generators import numbered_sentence, sentence

TOKENIZERS = [tokenize_on_spaces, tokenize_characters]
ALPHABET = (
    string.ascii_letters + string.digits + string.punctuation.replace("\\", "")
)

WORD_LENGTH = 20
VOCAB_LENGTH = 15
SENTENCE_LENGTH = 10

# Nomenclature:
# - a generator is a sampler that is meant to be reused in multiple places,
#   but not directly as a strategy for a test.
# - a strategy is a sampler that is bespoke to a function test.
#   Strategies can output multiple objects.

# ---- standard generators -----


# ---- Testing numericalization ----


@composite
def sentences(
    draw: DrawFn, min_length: int = 1, max_length: int = SENTENCE_LENGTH
) -> list[str]:
    """Generate a list of unique sentences."""
    return draw(
        st.lists(
            sentence(),
            min_size=min_length,
            max_size=max_length,
            unique=True,
        )
    )


@given(method=st.sampled_from(TOKENIZERS))
def test_numericalize(method: TokenizeFunction) -> None:
    """Test _numericalize.

    - Test whether sentences are converted to numpy.ndarrays
    - Test if the same number of sentences are numericalised as are entered in the numericalisation
    """  # noqa: E501
    sentences = [
        "Dit is een test",
        "Dit is een andere test",
        "Wat een hoop test",
    ]
    vocab = create_default_vocab(sentences, method)
    translator = Translator(method, vocab)
    numericalized = _numericalize(sentences, translator)
    assert all(isinstance(sentence, np.ndarray) for sentence in numericalized)
    assert len(numericalized) == len(sentences)


# ---- Testing weight matrix -----


@given(
    size=st.integers(min_value=2, max_value=10),
    base=st.floats(min_value=0.01, max_value=0.99),
)
def test_weight_matrix(size: int, base: float) -> None:
    """Test _weight_matrix.

    - Test if the weight matrix has the desired size
    - Test if the highest value in the matrix is 1
    - Test if the matrix is symmetric along the diagonal
    - Test if the diagonal is all ones
    """
    weight_matrix = _weight_matrix(size, base)

    assert weight_matrix.shape == (size, size)
    assert weight_matrix.max() == 1.0
    assert weight_matrix.min() == base ** (size - 1)
    assert np.array_equal(weight_matrix.T, weight_matrix)
    assert all(np.diag(weight_matrix))


@given(
    size=st.integers(min_value=2, max_value=10),
    # generating floats outside of the range [0, 1]
    base=st.floats(max_value=0),
)
def test_weight_matrix_exception(size: int, base: float) -> None:
    """Test if _weight_matrix() throws correct exception.

    - Test if the weight matrix throws an exception if the min value is outside the range [0, 1]
    """  # noqa: E501
    with pytest.raises(InvalidWeightBaseError):
        _weight_matrix(size, base)


@given(size=st.integers(min_value=2, max_value=10))
def test_weight_matrix_identity(size: int) -> None:
    """Test _weight_matrix with identity setting.

    - Test if the weight matrix has the correct shape when identity is set to True
    """
    weight_matrix = _weight_matrix(size, 1, identity=True)
    assert np.array_equal(weight_matrix, np.eye(size))


# ---- Testing one-hot encoding ----


@given(sentence=numbered_sentence(VOCAB_LENGTH, SENTENCE_LENGTH))
def test_one_hot_sentence(sentence: tuple[int]) -> None:
    """Test _one_hot_sentence.

    - Test if the one-hot-encoded matrix has dimensions of (vocab_length, max_sentence_length)
    - Test if the correct index in each column is one-hot-encoded
    """  # noqa: E501
    num_sentence = np.asarray(sentence)
    vocab_length = (
        num_sentence.max() + 1
    )  # making sure that if the sentence is [0] this is interpreted as a length of 1
    max_sentence_length = num_sentence.size

    wm = _weight_matrix(max_sentence_length, 0.9)

    encoded, _ = _one_hot_sentence(
        num_sentence, vocab_length, max_sentence_length, wm
    )

    assert encoded.shape == (vocab_length, max_sentence_length)
    assert np.array_equal(num_sentence, encoded.argmax(axis=0))


# ---- Testing einsum ----


@composite
def numbered_sentences_generator(
    draw: DrawFn,
    sentence_length: int = SENTENCE_LENGTH,
    vocab_length: int = VOCAB_LENGTH,
) -> list[np.ndarray]:
    """Generate a list of 'numbered_sentence's."""
    size = sentence_length
    sentences = draw(
        st.lists(
            numbered_sentence(vocab_length - 1, size),
            min_size=2,
            max_size=7,
            unique=True,
        )
    )
    return [np.asarray(s) for s in sentences]


@dataclass
class EinsumStrategy:
    """Container for Einsum strategy."""

    sentences: list[np.ndarray]
    vocab_length: int
    max_sentence_length: int


@composite
def einsum_strategy(draw: DrawFn) -> EinsumStrategy:
    """Generate a list of numbered sentences, a vocab length and a sentence length for testing einsum calculations."""  # noqa: E501
    sentence_length = draw(st.integers(min_value=5, max_value=10))
    vocab_length = draw(st.integers(min_value=sentence_length + 1, max_value=20))
    sentences = draw(
        numbered_sentences_generator(
            sentence_length=sentence_length, vocab_length=vocab_length
        )
    )
    return EinsumStrategy(sentences, vocab_length, sentence_length)


# somewhat more of an integeration test
# but generating the right tensors is really annoying
@given(einsum_strategy())
def test_einsum(s: EinsumStrategy) -> None:
    """Test _einsum.

    Pseudo-integration test of einsum
    - Test whether the einsum result has the correct shape
    - Test whether the diagonal has all ones
    - Test whether the minimum value is not smaller than 0
    """
    wm = _weight_matrix(s.max_sentence_length, 0.9)

    one_hot_encodings = [
        _one_hot_sentence(sentence, s.vocab_length, s.max_sentence_length, wm)
        for sentence in s.sentences
    ]
    one_hot_sentences, norms = list(zip(*one_hot_encodings, strict=False))
    norms = np.expand_dims(norms, axis=0)
    tensor = np.stack(one_hot_sentences)

    einsum = _einsum(tensor, np.asarray(norms), wm)

    n = len(s.sentences)
    assert einsum.shape == (n, n)
    assert all(np.diag(einsum))
    assert einsum.min() >= 0


# ---- Testing to_dataframe ----


@composite
def dataframe_strategy(
    draw: DrawFn,
) -> tuple[list[str], np.ndarray, bool]:
    """Generate fake einsum output that can be converted to a dataframe.

    Allows testing whether the right values also end up in the right places
    """
    sents = draw(sentences(min_length=2, max_length=4))
    # Generating a fake einsum matrix
    data = draw(
        arrays(
            dtype=np.float64,
            shape=(len(sents), len(sents)),
            elements=st.floats(min_value=0, max_value=1),
            unique=True,
        )
    )
    identity = draw(st.booleans())
    # Rounding to avoid floating point errors
    return sents, np.round(data, 5), identity


@given(dataframe_strategy())
def test_to_dataframe(data: tuple[list[str], np.ndarray, bool]) -> None:
    """Test _to_dataframe().

    - Test whether the correct columns are present
    - Test whether all the sentences are present in both columns
    - Test whether similarity scores end up in the correct place
    """
    sentences, similarity, identity = data

    sdf = _to_dataframe(sentences, similarity, filter_identity=identity)

    assert set(sdf.columns) == {"sentence", "other_sentence", "similarity"}
    assert set(sentences) == set(sdf["sentence"].unique())
    assert set(sentences) == set(sdf["other_sentence"].unique())

    assert isinstance(sdf, pl.DataFrame)

    # checking whether values have ended up in the right place
    indices = list(range(len(sentences)))
    for i, j in itertools.product(indices, indices):
        if i == j and identity:
            continue
        sentence = sentences[i]
        other_sentence = sentences[j]
        assert (
            sdf.filter(
                (pl.col("sentence") == pl.lit(sentence))
                & (pl.col("other_sentence") == pl.lit(other_sentence))
            )["similarity"].item()
            == similarity[i, j]
        )


# ---- integration test of sentence_similarity ----


@given(
    tokenizer=st.sampled_from(TOKENIZERS),
    weight_matrix_base=st.floats(min_value=0.01, max_value=0.999),
    filter_identity=st.booleans(),
)
def test_sentence_similarity(
    tokenizer: TokenizeFunction,
    weight_matrix_base: float,
    *,
    filter_identity: bool,
) -> None:
    """Test sentence_similarity.

    Integration test of sentence similarity:
    # - Test whether each sentence is in both [sentence] and [other_sentence] columns
    # - Test whether the provided sentences all show up in the [sentence] column
    # - Test whether the output is of the correct type (pd.DataFrame or pl.DataFrame)
    """
    sentences = [
        "Dit is een test",
        "Dit is een andere test",
        "Wat een hoop test",
    ]
    similarity = sentence_similarity(
        sentences,
        tokenizer,
        weight_matrix_base=weight_matrix_base,
        filter_identity=filter_identity,
    )

    assert isinstance(similarity, pl.DataFrame)

    assert set(similarity["sentence"].unique()) == set(
        similarity["other_sentence"].unique()
    )
    assert set(similarity["sentence"].unique()) == set(sentences)


def test_sentence_similarity_outcomes() -> None:
    """Test whether the values that come out of sentence similarity make sense."""
    sentences = ["bijna hetzelfde 1", "bijna hetzelfde 2"]
    out_df = pl.DataFrame(
        {
            "sentence": sentences,
            "other_sentence": sentences[::-1],
            "similarity": [2 / 3, 2 / 3],
        }
    )
    ss = sentence_similarity(sentences)
    assert_frame_equal(ss, out_df, check_row_order=False, check_column_order=False)


def test_sentence_similarity_raises_duplicate() -> None:
    """Test whether sentence similarity raises the correct exception."""
    sentences = ["exact hetzelfde", "exact hetzelfde", "bijna hetzelfde"]
    with pytest.raises(DuplicateSentencesError):
        sentence_similarity(sentences)
