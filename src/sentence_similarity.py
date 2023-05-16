import numpy as np
import pandas as pd
from functools import partial

from src.vocab import Vocab
    
def sentence_similarity(
        sentences: list[str],
        tokenize_method: str='words',
        lower: bool=False,
        term_frequency: bool=True,  # ? perhaps remove option if it is always better to do or not do
        scale_by_length: bool=False, # ? perhaps remove option if it is always better to do or not do
        ) -> pd.DataFrame:

    # Creating vocabulary to translate sentences into numbers
    vocab = Vocab(sentences, tokenize_method, lower)
    vocab_length = len(vocab)
    # Numericalizing sentences
    num_sentences = _numericalize(sentences, vocab)
    max_sentence_length = max(len(sentence) for sentence in num_sentences)
    # One-hot encoding 
    one_hot_kwargs = dict(vocab_length=vocab_length, 
                          max_sentence_length=max_sentence_length, 
                          term_frequency=term_frequency, 
                          scale_by_length=scale_by_length)
    one_hot_encode = partial(_one_hot_sentence, **one_hot_kwargs)
    one_hot_sentences = [one_hot_encode(sentence) for sentence in num_sentences]

    pass

def _numericalize(sentences: list[str], vocab: Vocab) -> list[np.ndarray]:
    return [np.asarray(vocab.encode(sentence)) for sentence in sentences]

def _one_hot_sentence(
        sentence: np.ndarray, 
        vocab_length: int, 
        max_sentence_length: int,
        term_frequency: bool,
        scale_by_length: bool,
        ) -> np.ndarray:
    # starting with a matrix with a row for each word in the vocab, and a column for each possible word in a sentence
    one_hot = np.zeros((vocab_length, max_sentence_length))
    # replacing indices given from the numericalised sentences with ones
    one_hot[(sentence, np.arange(len(sentence)))] = 1.
    # Index 0 indicates that the token was not recognised by the vocabulary. These need to be ignored in the similarity check,
    # so setting the first row to zeros:
    one_hot[0] = np.zeros((1, max_sentence_length))

    # when term frequency is applied, the value of a word appearing is divided by how often it appears in the sentence.
    # This avoids getting high similarity scores because the same word appears mulitple times in the sentence.
    if term_frequency:
        term_frequencies = one_hot.sum(axis=1) # counting how often each word appears by summing over the columns
        inverse_term_frequencies = np.divide(1, term_frequencies, where=term_frequencies > 0) # avoiding divide by zero error with np.divide(where=...)
        # multiply row by inverse term frequencies
        one_hot = np.einsum("ij, i -> ij", one_hot, inverse_term_frequencies, optimize="optimal")
    
    if scale_by_length:
        one_hot = one_hot * 1/len(sentence)

    return one_hot
