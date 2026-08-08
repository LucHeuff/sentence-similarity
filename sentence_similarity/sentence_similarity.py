"""Contains sentence similarity main function and algorithm."""

from functools import partial

import numpy as np
import polars as pl

from sentence_similarity.exceptions import (
    DuplicateSentencesError,
    InvalidWeightBaseError,
)
from sentence_similarity.translator import (
    TokenizeFunction,
    Translator,
    create_default_vocab,
    tokenize_words,
)

EINSUM_OPT = "optimal"


def sentence_similarity(
    sentences: list[str],
    tokenizer: TokenizeFunction = tokenize_words,
    translator: Translator | None = None,
    *,
    weight_matrix_base: float = 0.9,
    weight_matrix_identity: bool = False,
    filter_identity: bool = True,
) -> pl.DataFrame:
    """Calculate similarity among provided sentences.

    Args:
    ----
        sentences: list of sentences to be compared to each other
        tokenizer: tokenisation function, allows passing in custom tokenizer.
                   Defaults to tokenize_words.
        translator: Translator object, allows passing in custom instance.
                    Generates default translator when left empty.
        weight_matrix_base: Scaling for weight matrix, performs base ^ distance.
                            Ignored if weight_matrix_identity=True
        weight_matrix_identity: sets the weight matrix to the identity matrix.
                                This means you are only interested to what extent
                                sentences have the same words in the same place.

        filter_identity: whether cases where the sentence is compared to itself
                         should be filtered out.

    Returns:
    -------
        A dataframe with columns (sentence, other_sentence, score) containing
        the paired sentences and the calculated similarity score.
        A score of 1 indicates the sentences are the same.
        A score of 0 indicates the sentences have nothing in common.
        A score between 0 and 1 is a measure of the similarity between two sentences

    """
    if len(sentences) != len(set(sentences)):
        message = "[sentences] is not unique, some sentences are duplicated."
        raise DuplicateSentencesError(message)

    # Creating vocabulary to translate sentences into numbers
    if translator is None:
        vocab = create_default_vocab(sentences, tokenizer)
        translator = Translator(tokenizer, vocab)

    vocab_length = len(translator)

    num_sentences = _numericalize(sentences, translator)
    max_sentence_length = max(len(sentence) for sentence in num_sentences)

    weight_matrix = _weight_matrix(
        max_sentence_length, weight_matrix_base, identity=weight_matrix_identity
    )

    one_hot_encode = partial(
        _one_hot_sentence,
        vocab_length=vocab_length,
        max_sentence_length=max_sentence_length,
        weight_matrix=weight_matrix,
    )

    one_hot_encodings = [one_hot_encode(sentence) for sentence in num_sentences]
    one_hot_sentences, norms = list(zip(*one_hot_encodings, strict=False))

    # Converting norms to a 1D array so it plays nicely with transpostions
    norms = np.expand_dims(norms, axis=0)

    one_hot_tensor = np.stack(one_hot_sentences)

    similarity = _einsum(one_hot_tensor, norms, weight_matrix)

    return _to_dataframe(
        sentences,
        similarity,
        filter_identity=filter_identity,
    )


def _numericalize(sentences: list[str], translator: Translator) -> list[np.ndarray]:
    return [np.asarray(translator.encode(sentence)) for sentence in sentences]


def _one_hot_sentence(
    sentence: np.ndarray,
    vocab_length: int,
    max_sentence_length: int,
    weight_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convert a numericalised sentence into a matrix of one-hot encodings.

    Rows represent tokens in the vocabulary, columns represent the word order.

    Args:
    ----
        sentence: sentence numericalised through the vocabulary
        vocab_length: total number of tokens in the vocabulary
        max_sentence_length: length of the longest sentence in the comparison.
                                 This makes sure all matrices are of the same size,
                                 padded with zeros when shorter.
        weight_matrix: to appropriately scale the norm

    Returns:
    -------
        tuple of:
        np.ndarray: Matrix one-hot-encoding of the sentence
        float: norm of the sentence


    """
    index = np.arange(len(sentence))

    # building the one-hot matrix
    one_hot = np.zeros((vocab_length, max_sentence_length))
    one_hot[sentence, index] = 1

    norm = ((one_hot.T @ one_hot) * weight_matrix).sum()

    return one_hot, norm


def _weight_matrix(size: int, base: float, *, identity: bool = False) -> np.ndarray:
    """Generate a weight matrix.

    The weight matrix is used to discount cases where words do appear in a pair of sentences,
    but in different positions in the sentence. This matrix has 1.'s on the diagonal, and reduce
    out the the provided minimum value to the top-right and bottom-left corners of the matrix.

    Args:
    ----
        size: Determines the shape of the square matrix (size, size)
        base: for the weight scaling. Performs base ^ distance. Ignored when identity=True
        identity: sets the weight matrix to an identity matrix of shape (size, size)

    Returns:
    -------
        the weight matrix

    Raises:
    ------
        ValueError: When [base] is not larger than 0


    """  # noqa: E501
    if identity:
        return np.eye(size)

    if base <= 0:
        msg = f"Base should be larger than 0, but got {base}"
        raise InvalidWeightBaseError(msg)

    i = np.asarray([np.arange(size)])
    distance = np.abs(i.T - i)
    weight = base**distance
    assert weight.shape == (size, size), "Weight matrix has incorrect shape."

    return weight


def _einsum(
    tensor: np.ndarray, norms: np.ndarray, weight_matrix: np.ndarray
) -> np.ndarray:
    """Calculate similarity among sentences.

    Similarity is calculated by performing tensor inner product,
    scaling by the weight matrix, and scaling the result over the diagonal.
    This is implemented using np.einsum to vastly optimise the calculation time

    Args:
    ----
        tensor: encoded sentences in tensor form,
                of shape (sentences, vocabulary length, max sentence length)
        norms: array containing norm of each sentence
        weight_matrix: weight matrix of shape (sentences, sentences)

    Returns:
    -------
        np.ndarray: similarity scores of shape (sentences, sentences)

    """
    # should result in norm product for each sentence
    norm = 1 / np.sqrt(norms.T @ norms)

    return np.einsum(
        "mij, nik, jk, mn -> mn",
        tensor,
        tensor,
        weight_matrix,
        norm,
        optimize=EINSUM_OPT,
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
        max_scores: number of tokens in each sentence
        similarity: output of _einsum()
        filter_identity: whether to remove rows where sentence is equal to other_sentence. Default: True.

    Returns:
    -------
        pd.DataFrame: dataframe consisting of sentence pair columns named 'sentence'
                    and 'other_sentence' with their similarity score in 'similarity'

    """  # noqa: E501
    sdf = (
        pl.from_numpy(similarity, schema=sentences)
        .with_columns(sentence=pl.Series(sentences))
        .unpivot(
            index="sentence", variable_name="other_sentence", value_name="similarity"
        )
        # rounding for floating point stuff
        .with_columns(pl.col("similarity").round(5))
    )

    if filter_identity:
        sdf = sdf.filter(pl.col("sentence") != pl.col("other_sentence"))

    return sdf
