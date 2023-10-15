from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd

from src.translator import (
    TokenizeFunction,
    create_default_translator,
    tokenize_on_spaces,
    tokenize_words,
)


class Translator(Protocol):
    """Protocol class for translator methods"""

    def encode(self, sentence: str) -> list[int]:
        """Translates a string sentence into a list of integers"""
        ...

    def __len__(self) -> int:
        """Returns the length of the vocabulary of the translator"""
        ...


EINSUM_OPT = "optimal"


def sentence_similarity(
    sentences: list[str],
    tokenizer: TokenizeFunction = tokenize_words,
    translator: Translator | None = None,
    weight_matrix_min: float = 0.1,
) -> pd.DataFrame:
    """Compares sentences in the form of strings through a tokenisation method.

    Args:
        sentences (list[str]): list of sentences to be compared to each other in the form of strings
        tokenizer (TokenizeFunction, optional): function to perform tokenization.
                 Also allows providing custom tokenization function. Defaults to tokenize_on_spaces.
        translator (Translator | None, optional): Translator object that performs sentence encoding.
                                 Also allows providing a custom Translator object. Defaults to None.
        weight_matrix_min (float, optional): The weight matrix discounts sentences that have the
            same words, but in different places. This value controls the weight of the value that
            is furthest out. You may wish to raise the value if using short sentences or a small
            vocabulary. Defaults to 0.1.

    Returns:
        pd.DataFrame: containing each possible sentence pair and the corresponding similarity score.
                      A similarity score of 1 indicates the sentences are the same. A score below 1
                      means indicates a reduction in similarity. A score above 1 indicates that
                      substrings in the sentence are repeated multiple times in the other sentence,
                      which tends to happen more often when tokenizing characters.
    """
    # Creating vocabulary to translate sentences into numbers
    if translator is None:
        translator = create_default_translator(sentences, tokenizer)
    vocab_length = len(translator)

    num_sentences = _numericalize(sentences, translator)
    max_sentence_length = max(len(sentence) for sentence in num_sentences)

    one_hot_encode = partial(
        _one_hot_sentence,
        vocab_length=vocab_length,
        max_sentence_length=max_sentence_length,
    )
    one_hot_sentences = [one_hot_encode(sentence) for sentence in num_sentences]

    one_hot_tensor = np.stack(one_hot_sentences)
    weight_matrix = _weight_matrix(max_sentence_length, weight_matrix_min)

    similarity = _einsum(one_hot_tensor, weight_matrix)

    return _to_dataframe(sentences, similarity)


def _numericalize(sentences: list[str], translator: Translator) -> list[np.ndarray]:
    return [np.asarray(translator.encode(sentence)) for sentence in sentences]


def _one_hot_sentence(
    sentence: np.ndarray,
    vocab_length: int,
    max_sentence_length: int,
) -> np.ndarray:
    """Converts a numericalised sentence (e.g. [1, 2, 3]) into a matrix of one-hot encodings.
    Rows represent tokens in the vocabulary, columns represent the word order.
    Term frequency reduction is applied to handle cases where words are repeated in the sentence.

    Args:
        sentence (np.ndarray): sentence numericalised through the vocabulary
        vocab_length (int): total number of tokens in the vocabulary
        max_sentence_length (int): length of the longest sentence in the comparison.
                                 This makes sure all matrices are of the same size,
                                 padded with zeros when shorter.

    Returns:
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
    term_frequencies = one_hot.sum(
        axis=1
    )  # counting how often each word appears by summing over the columns
    inverse_term_frequencies = np.divide(
        1, np.sqrt(term_frequencies), where=term_frequencies > 0
    )  # avoiding divide by zero error with np.divide(where=...)
    # multiply row by inverse term frequencies
    one_hot = np.einsum(
        "ij, i -> ij", one_hot, inverse_term_frequencies, optimize=EINSUM_OPT
    )
    one_hot = np.nan_to_num(
        one_hot
    )  # making sure there are no nans in my array which seems to happen in testing

    return one_hot


def _weight_matrix(size: int, minimum: float) -> np.ndarray:
    """Generates a weight matrix to discount cases where words do appear in a pair of sentences,
    but in different positions in the sentence. This matrix has 1.'s on the diagonal, and reduce
    out the the provided minimum value to the top-right and bottom-left corners of the matrix.

    Args:
        size (int): Determines the shape of the square matrix (size, size)
        minimum (float): Determines the minimum value to enter into the weight matrix.
                         Should be between 0. and 1. Setting this to 1 disables weight scaling.

    Raises:
        ValueError: When [minimum] is not between 0. and 1.

    Returns:
        np.ndarray: weight matrix of shape (size, size) with 1. on diagonal
                    and lowering to [minimum] on towards the edges.
    """

    if not 0 <= minimum <= 1:
        raise ValueError(
            f"""You are trying to set a minimum value for the weight matrix of {minimum},
            which is outside the range [0., 1.].
            The weight matrix was not designed with this in mind"""
        )

    size_range = np.arange(size)
    linear_space = np.linspace(1, minimum, num=size)

    weights = sum(np.eye(size, k=n) * s for n, s in zip(size_range, linear_space))
    weight_matrix = np.triu(weights) + np.triu(weights).T - np.eye(size)

    return weight_matrix


def _einsum(tensor: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """Calculates similarity among sentences by performing tensor inner product,
    scaling by the weight matrix, and scaling the result over the diagonal.
    This is implemented using np.einsum to vastly optimise the calculation time

    Args:
        tensor (np.ndarray): encoded sentences in tensor form,
                             of shape (sentences, vocabulary length, max sentence length)
        weight_matrix (np.ndarray): weight matrix of shape (sentences, sentences)

    Returns:
        np.ndarray: similarity scores of shape (sentences, sentences)
    """
    # einsum to calculate the similarity scores for all sentences amongst each other
    similarity = np.einsum(
        "mij, nik, jk -> mn", tensor, tensor, weight_matrix, optimize=EINSUM_OPT
    )

    # scaling down columnwise by diagonal, this should result in
    # scoring in a column being relative to the sentence itself
    similarity = np.einsum(
        "ij, j -> ij", similarity, 1 / np.diag(similarity), optimize=EINSUM_OPT
    )

    return similarity


def _to_dataframe(sentences: list[str], similarity: np.ndarray) -> pd.DataFrame:
    """Constructs a pandas.DataFrame containing the combination of
    each pair of sentences with their similarity scores.

    Args:
        sentences (list[str]): list of sentences that were compared to each other using _einsum()
        similarity (np.ndarray): output of _einsum()

    Returns:
        pd.DataFrame: dataframe consisting of sentence pair columns named 'sentence'
                    and 'other_sentence' with their similarity score in 'similarity'
    """

    new_names = {"level_0": "sentence", "level_1": "other_sentence", 0: "similarity"}
    df = (
        pd.DataFrame(similarity, index=sentences, columns=sentences)
        .stack()
        .reset_index()
        .rename(columns=new_names)
    )
    return df

