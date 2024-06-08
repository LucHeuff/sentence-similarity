"""Contains sentence similarity main function and algorithm."""
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd

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

    return _to_dataframe(sentences, similarity)


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
    # starting with a matrix of zeros with a row for each word in the vocab,
    # and a column for each possible word in a sentence
    one_hot = np.zeros((vocab_length, max_sentence_length))
    # replacing indices given from the numericalised sentences with ones
    one_hot[(sentence, np.arange(len(sentence)))] = 1.0

    # Term frequency is applied to discount words that appear in the sentence more often.
    # This avoids getting high similarity scores because the same word appears mulitple
    # times in the sentence. Discounting with 1/sqrt(n), so my similarity score doesn't
    # disappear too much simply through repeated words

    # counting how often each word appears by summing over the columns
    term_frequencies = one_hot.sum(axis=1)

    # avoiding divide by zero error with np.divide(where=...)
    inverse_term_frequencies = np.divide(
        1, np.sqrt(term_frequencies), where=term_frequencies > 0
    )
    # multiply row by inverse term frequencies
    one_hot = np.einsum(
        "ij, i -> ij", one_hot, inverse_term_frequencies, optimize=EINSUM_OPT
    )
    # making sure there are no nans in my array which seems to happen in testing
    return np.nan_to_num(one_hot)


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


def _einsum(tensor: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """Calculate similarity among sentences.

    Similarity is calculated by performing tensor inner product,
    scaling by the weight matrix, and scaling the result over the diagonal.
    This is implemented using np.einsum to vastly optimise the calculation time

    Args:
    ----
        tensor (np.ndarray): encoded sentences in tensor form,
                             of shape (sentences, vocabulary length, max sentence length)
        weight_matrix (np.ndarray): weight matrix of shape (sentences, sentences)

    Returns:
    -------
        np.ndarray: similarity scores of shape (sentences, sentences)

    """
    # einsum to calculate the similarity scores for all sentences amongst each other
    similarity = np.einsum(
        "mij, nik, jk -> mn", tensor, tensor, weight_matrix, optimize=EINSUM_OPT
    )

    # scaling down columnwise by diagonal, this should result in
    # scoring in a column being relative to the sentence itself
    return np.einsum(
        "ij, j -> ij", similarity, 1 / np.diag(similarity), optimize=EINSUM_OPT
    )


def _to_dataframe(sentences: list[str], similarity: np.ndarray) -> pd.DataFrame:
    """Construct a pandas.DataFrame containing the combination of each pair of sentences with their similarity scores.

    Args:
    ----
        sentences (list[str]): list of sentences that were compared to each other using _einsum()
        similarity (np.ndarray): output of _einsum()

    Returns:
    -------
        pd.DataFrame: dataframe consisting of sentence pair columns named 'sentence'
                    and 'other_sentence' with their similarity score in 'similarity'

    """
    new_names = {
        "level_0": "sentence",
        "level_1": "other_sentence",
        0: "similarity",
    }
    return (
        pd.DataFrame(similarity, index=sentences, columns=sentences)  # type: ignore
        .stack()
        .reset_index()
        .rename(columns=new_names)
    )
