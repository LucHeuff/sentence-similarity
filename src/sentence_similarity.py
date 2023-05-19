import numpy as np
import pandas as pd
from functools import partial

from src.vocab import Vocab
    
def sentence_similarity(
        sentences: list[str],
        tokenize_method: str='words',
        lower: bool=False,
        weight_matrix_min: float=0.1, # ? perhaps remove option if it is always better to do or not do
        ) -> pd.DataFrame:

    # Creating vocabulary to translate sentences into numbers
    vocab = Vocab(sentences, tokenize_method, lower)
    vocab_length = len(vocab)

    num_sentences = _numericalize(sentences, vocab)
    max_sentence_length = max(len(sentence) for sentence in num_sentences)

    one_hot_encode = partial(_one_hot_sentence, vocab_length=vocab_length, max_sentence_length=max_sentence_length)
    one_hot_sentences = [one_hot_encode(sentence) for sentence in num_sentences]

    one_hot_tensor = np.stack(one_hot_sentences)
    weight_matrix = _weight_matrix(max_sentence_length, weight_matrix_min)

    similarity = _einsum(one_hot_tensor, weight_matrix)

    return _to_dataframe(sentences, similarity)

def _half_precision(array: np.ndarray) -> np.ndarray:
    return array.astype(dtype=np.float16)

def _numericalize(sentences: list[str], vocab: Vocab) -> list[np.ndarray]:
    return [np.asarray(vocab.encode(sentence)) for sentence in sentences]

def _one_hot_sentence(
        sentence: np.ndarray, 
        vocab_length: int, 
        max_sentence_length: int,
        ) -> np.ndarray:
    # starting with a matrix of zeros with a row for each word in the vocab, and a column for each possible word in a sentence
    one_hot = np.zeros((vocab_length, max_sentence_length))
    # replacing indices given from the numericalised sentences with ones
    one_hot[(sentence, np.arange(len(sentence)))] = 1.

    # Term frequency is applied to discount words that appear in the sentence more often. 
    # This avoids getting high similarity scores because the same word appears mulitple times in the sentence.
    # Discounting with 1/sqrt(n), so my similarity score doesn't disappear too much simply through repeated words
    term_frequencies = one_hot.sum(axis=1) # counting how often each word appears by summing over the columns
    inverse_term_frequencies = np.divide(1, np.sqrt(term_frequencies), where=term_frequencies > 0) # avoiding divide by zero error with np.divide(where=...)
    # multiply row by inverse term frequencies
    one_hot = np.einsum("ij, i -> ij", one_hot, inverse_term_frequencies, optimize="optimal")
    one_hot = np.nan_to_num(one_hot) # making sure there are no nans in my array which seems to happen in testing

    return _half_precision(one_hot)  # * using half precision floats to save some space

def _weight_matrix(size: int, min: float) -> np.ndarray:

    if min >= 1.0:
        raise ValueError(f"You are trying to set a minimum value of {min}, which is higher or equal to 1. The weight matrix was not designed with this in mind")

    size_range = np.arange(size)
    linear_space = np.linspace(1, min, num=size)

    weights = sum(np.eye(size, k=n) * s for n, s in zip(size_range, linear_space))
    weight_matrix = np.triu(weights) + np.triu(weights).T - np.eye(size)
    
    return _half_precision(weight_matrix)

def _einsum(tensor: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    # einsum to calculate the similarity scores for all sentences amongst each other
    similarity = np.einsum("mij, nik, jk -> mn", tensor, tensor, weight_matrix, optimize="optimal")

    # scaling down columnwise by diagonal, this should result in scoring in a column being relative to the sentence itself
    similarity = np.einsum("ij, j -> ij", similarity, 1 / np.diag(similarity), optimize="optimal")

    # TODO why was this necessary? -> try without first
    # taking lower triangle, as similarity should logically be symmetric
    # similarity = np.tril(similarity) + np.tril(similarity).T - np.diag(np.diag(similarity))

    return similarity

def _to_dataframe(sentences: list[str], similarity: np.ndarray) -> pd.DataFrame:
    new_names = {'level_0': 'sentence', 'level_1': 'other_sentence', 0: 'similarity'}
    df = (pd.DataFrame(similarity, index=sentences, columns=sentences)
          .stack()
          .reset_index()
          .rename(columns=new_names)
          )
    return df

    # TODO add scaling down cases where similarity larger than 1 through 1 - ([sim>1] - 1) before dividing by diagonal -> in the pandas conversion!