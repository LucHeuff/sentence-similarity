"""Contains sentence similarity main function and algorithm."""
from collections import Counter
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl

from sentence_similarity.translator import (
    TokenizeFunction,
    create_default_translator,
    tokenize_words,
)


class Translator(Protocol):
    """Protocol class for translator methods."""

    def encode(self, sentence: str) -> list[int]:
        """Translate a string sentence into a list of integers."""
        ...

    def __len__(self) -> int:
        """Return the length of the vocabulary of the translator."""
        ...


EINSUM_OPT = "optimal"


def sentence_similarity(
    sentences: list[str],
    tokenizer: TokenizeFunction = tokenize_words,
    translator: Translator | None = None,
    weight_matrix_min: float | str = 0.1,
    *,
    filter_identity: bool = True,
) -> pd.DataFrame:
    """Calculate similarity among provided sentences.

    Args:
    ----
        sentences: list of sentences to be compared to each other
        tokenizer: function that performs tokenisation, allows passing in custom tokenizer.
                   Defaults to tokenize_words.
        translator: Translator object, allows passing in custom instance. Generates default
                    translator when left empty. Defaults to None.
        weight_matrix_min: allows setting the extreme values of the weight matrix, which
                           discounts words that match between sentences but are not in the
                           same position. If float, must be between 0 and 1.
                           Set to 'identity' if you want to ignore words that are not in
                           the correct position entirely.
        filter_identity: whether cases where the sentence is compared to itself should be filtered out.

    Returns:
    -------
        A dataframe with columns (sentence, other_sentence, score) containing the paired
        sentences and the calculated similarity score.
        A score of 1 indicates the sentences are the same.
        A score of 0 indicates the sentences have nothing in common.
        A score between 0 and 1 is a measure for the similarity between the two sentences
        A score larger than 1 indicates that some tokens are repeated in one or both sentences.

    Raises:
    ------
        ValueError: if a string is passed into weight_matrix_min that is not 'identity'.

    """
    # Creating vocabulary to translate sentences into numbers
    if translator is None:
        translator = create_default_translator(sentences, tokenizer)
    vocab_length = len(translator)

    num_sentences = _numericalize(sentences, translator)
    max_sentence_length = max(len(sentence) for sentence in num_sentences)

    # creating weight matrix as soon as possible since it might raise an exception
    if weight_matrix_min == "identity":
        weight_matrix = _weight_matrix(max_sentence_length, identity=True)
    elif isinstance(weight_matrix_min, str):
        raise ValueError(
            f"weight_matrix_min should be float or 'identity', got {weight_matrix_min}"
        )
    else:
        weight_matrix = _weight_matrix(max_sentence_length, weight_matrix_min)

    one_hot_encode = partial(
        _one_hot_sentence,
        vocab_length=vocab_length,
        max_sentence_length=max_sentence_length,
    )
    one_hot_sentences = [one_hot_encode(sentence) for sentence in num_sentences]

    one_hot_tensor = np.stack(one_hot_sentences)

    similarity = _einsum(one_hot_tensor, weight_matrix)

    return _to_dataframe(sentences, similarity, filter_identity=filter_identity)


def _numericalize(
    sentences: list[str], translator: Translator
) -> list[np.ndarray]:
    return [np.asarray(translator.encode(sentence)) for sentence in sentences]


def _one_hot_sentence(
    sentence: np.ndarray,
    vocab_length: int,
    max_sentence_length: int,
) -> np.ndarray:
    """Convert a numericalised sentence (e.g. [1, 2, 3]) into a matrix of one-hot encodings.

    Rows represent tokens in the vocabulary, columns represent the word order.
    Term frequency reduction is applied to handle cases where words are repeated in the sentence.

    Args:
    ----
        sentence (np.ndarray): sentence numericalised through the vocabulary
        vocab_length (int): total number of tokens in the vocabulary
        max_sentence_length (int): length of the longest sentence in the comparison.
                                 This makes sure all matrices are of the same size,
                                 padded with zeros when shorter.

    Returns:
    -------
        np.ndarray: Matrix encoding of the sentence. Note that when term frequencies are applied,
                    the matrix can take any value between 0. and 1.

    """
    # Determining the weight of words, in case they occur more than once.
    # Weighting by 1/sqrt(n) such that the similarity score still adds up to 1
    # even if words appear multiple times in the encoding products

    counter = Counter(sentence)
    values = [1 / np.sqrt(counter[word]) for word in sentence]

    index = np.arange(len(sentence))

    # building the one-hot matrix
    one_hot = np.zeros((vocab_length, max_sentence_length))

    # replacing indices given from the numericalised sentences with the require value
    one_hot[sentence, index] = values

    return one_hot


def _weight_matrix(
    size: int, minimum: float = 0.1, *, identity: bool = False
) -> np.ndarray:
    """Generate a weight matrix.

    The weight matrix is used to discount cases where words do appear in a pair of sentences,
    but in different positions in the sentence. This matrix has 1.'s on the diagonal, and reduce
    out the the provided minimum value to the top-right and bottom-left corners of the matrix.

    Args:
    ----
        size: Determines the shape of the square matrix (size, size)
        minimum: Determines the minimum value to enter into the weight matrix.
                 Should be between 0. and 1. Setting this to 1 disables weight scaling.
                 Defaults to 0.1.
        identity: sets the weight matrix to an identity matrix of shape (size, size)

    Returns:
    -------
        the weight matrix

    Raises:
    ------
        ValueError: When [minimum] is not between 0. and 1.


    """
    if identity:
        return np.eye(size)
    if not 0 <= minimum <= 1:
        raise ValueError(
            f"""You are trying to set a minimum value for the weight matrix of {minimum},
            which is outside the range [0., 1.].
            The weight matrix was not designed with this in mind"""
        )

    size_range = np.arange(size)
    linear_space = np.linspace(1, minimum, num=size)

    weights = sum(
        np.eye(size, k=n) * s for n, s in zip(size_range, linear_space)
    )
    weight_matrix = np.triu(weights) + np.triu(weights).T - np.eye(size)

    assert weight_matrix.shape == (
        size,
        size,
    ), "weight matrix has incorrect shape"

    return weight_matrix


def _einsum(
    tensor: np.ndarray,
    weight_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate similarity among sentences.

    Similarity is calculated by performing tensor inner product,
    scaling by the weight matrix, and scaling the result over the diagonal.
    This is implemented using np.einsum to vastly optimise the calculation time

    Args:
    ----
        tensor: encoded sentences in tensor form,
                             of shape (sentences, vocabulary length, max sentence length)
        weight_matrix: weight matrix of shape (sentences, sentences)

    Returns:
    -------
        np.ndarray: similarity scores of shape (sentences, sentences)

    """
    # einsum to calculate the similarity scores for all sentences amongst each other
    similarity = np.einsum(
        "mij, nik, jk -> mn",
        tensor,
        tensor,
        weight_matrix,
        optimize=EINSUM_OPT,
    )

    # scaling down columnwise by diagonal, this should result in
    # scoring in a column being relative to the sentence itself
    # This is needed because the weight matrix will also discount tokens that
    # are repeated in the same sentence. This step should bump these cases back
    # up to 1.
    return np.einsum(
        "ij, j -> ij", similarity, 1 / np.diag(similarity), optimize=EINSUM_OPT
    )


def _to_dataframe(
    sentences: list[str],
    similarity: np.ndarray,
    *,
    filter_identity: bool = True,
) -> pl.DataFrame:
    """Construct a pandas.DataFrame containing the combination of each pair of sentences with their similarity scores.

    Args:
    ----
        sentences: list of sentences that were compared to each other using _einsum()
        similarity: output of _einsum()
        filter_identity: whether to remove rows where sentence is equal to other_sentence. Default: True.

    Returns:
    -------
        pd.DataFrame: dataframe consisting of sentence pair columns named 'sentence'
                    and 'other_sentence' with their similarity score in 'similarity'

    """
    # polars variant

    df = (
        pl.from_numpy(similarity, schema=sentences)
        .with_columns(sentence=pl.Series(sentences))
        .melt(
            id_vars="sentence",
            variable_name="other_sentence",
            value_name="similarity",
        )
    )

    if filter_identity:
        return df.filter(pl.col("sentence") != pl.col("other_sentence"))
    return df

    new_names = {
        "level_0": "sentence",
        "level_1": "other_sentence",
        0: "similarity",
    }
    df = (
        pd.DataFrame(similarity, index=sentences, columns=sentences)  # type: ignore
        .stack()
        .reset_index()
        .rename(columns=new_names)
    )
    if filter_identity:
        df = df[~(df.sentence == df.other_sentence)]
    return df  # type: ignore
