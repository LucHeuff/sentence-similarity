"""Contains Translator, tokenizer and vocabulary functions to allow customisation of sentence similarity calculations."""
import re
from typing import Callable

import numpy as np
from strsimpy.levenshtein import Levenshtein
from strsimpy.string_distance import StringDistance

TokenizeFunction = Callable[[str], list[str]]

# ---- Creating basic tokenizers ----


def tokenize_words(sentence: str) -> list[str]:
    """Separate out tokens in sentence based on whitespaces.

    Will tokenize punctuation as a separate token, even if they are
    directly connected to a word.

    Args:
    ----
        sentence: string to be separated into tokens

    Returns:
    -------
        list of string tokens

    """
    sentence = re.sub(r"([^a-zA-Z\d\s])", "  \\1", sentence)
    return sentence.split()


def tokenize_on_spaces(sentence: str) -> list[str]:
    """Separate out tokens in the sentence based on whitespaces.

    Args:
    ----
        sentence: string to be separated into tokens

    Returns:
    -------
        list of string tokens

    """
    return sentence.split()


def tokenize_characters(sentence: str) -> list[str]:
    """Separate out each character as a separate token.

    Args:
    ----
        sentence: string to be separated into tokens

    Returns:
    -------
        list of string tokens

    """
    return list(sentence)


# ---- Translator ----


class Translator:
    """Base class for Translators."""

    def __init__(
        self, tokenizer: TokenizeFunction, vocab: dict[str, int]
    ) -> None:
        """Initialise a Translator.

        Args:
        ----
            tokenizer: a function that takes as argument a str and returns a list[str]
            vocab: a dictionary with str as keys and int as values.
                   Note that the lowest allowed value is 0!

        """
        self.tokenize = tokenizer
        self.vocab = vocab
        assert min(vocab.values()) == 0, "Lowest value in vocab should be 0."
        assert (
            len(vocab) == max(vocab.values()) + 1
        ), "The largest value in the vocab should be equal to the length of the vocab."

    def encode(self, sentence: str) -> list[int]:
        """Encode each word in the sentence, resulting in list of integers."""
        return [self.vocab[token] for token in self.tokenize(sentence)]

    def __len__(self) -> int:
        """Return the length of the Translator."""
        return (
            max(self.vocab.values()) + 1
        )  # Adding one since arrays start at 0


#  Translator factories
def create_default_translator(
    sentences: list[str], tokenizer: TokenizeFunction = tokenize_words
) -> Translator:
    """Create a default translator from a list of sentences and a tokenizer function.

    Args:
    ----
        sentences: list of sentences to be translated.
        tokenizer: function with which to perform
                                   tokenization. Defaults to tokenize_words.

    Returns:
    -------
        Translator: object to perform translation from sentences to numericalised lists.

    """
    vocab = create_vocab(sentences, tokenizer)
    return Translator(tokenizer, vocab)


def create_translator(
    vocab: dict[str, int], tokenizer: TokenizeFunction
) -> Translator:
    """Create a Translator, given a vocan and a tokenizer.

    Args:
    ----
        vocab: the desired vocab for the translator
        tokenizer: the desired tokenizer for the translator

    Returns:
    -------
        A Translator instance using the provided tokenizer and vocab

    """
    return Translator(tokenizer, vocab)


# ---- Functions for creating vocabs ----


def _tokenize_sentences(
    sentences: list[str], tokenizer: TokenizeFunction
) -> set[str]:
    """Convert sentences into a set of unique tokens.

    Args:
    ----
        sentences: sentences to be tokenized
        tokenizer: tokenizer function

    Returns:
    -------
        set[str]: set of unique tokens extracted from the sentences

    """
    # making sure each sentence is unique, so we don't do redundant operations
    sentences = list(set(sentences))
    # tokenizing each sentence and then flattening to a set of tokens
    tokenized_sentences = [tokenizer(sentence) for sentence in sentences]
    return {token for sentence in tokenized_sentences for token in sentence}


def create_vocab(
    sentences: list[str], tokenizer: TokenizeFunction
) -> dict[str, int]:
    """Create a vocabulary dictionary which pairs each unique token in the sentences with a unique integer.

    Args:
    ----
        sentences: list of sentences in the form of strings
        tokenizer: function that performs tokenization on sentences.

    Returns:
    -------
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values

    """
    # extracting unique tokens from sentences
    tokens = sorted(_tokenize_sentences(sentences, tokenizer))
    # enumerating these into a dictionary
    return {token: i for (i, token) in enumerate(tokens)}


def create_synonym_vocab(
    sentences: list[str],
    synonyms: list[tuple[str]],
    tokenizer: TokenizeFunction,
) -> dict[str, int]:
    """Create a vocabulary dictionary which pairs tokens to integers.

    Allows passing in lists of synonyms which translate to the same integer.

    Args:
    ----
        sentences (list[str]): list of sentences in the form strings
        synonyms (list[tuple[str]]): list of tuples, where each tuple is filled with
                                     all the words that are synonyms of each other
        tokenizer (tokenize_function): function that performs tokenization on sentences.

    Raises:
    ------
        ValueError: when there are tokens in a synonym tuple that do not appear in the sentences.

    Returns:
    -------
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values

    """
    # extracting unique tokens from sentences
    tokens = sorted(_tokenize_sentences(sentences, tokenizer))
    # flatting the list of synonym tuples so it's easier to check if tokens appear in them
    all_synonym_tokens = [
        synonym for synonym_list in synonyms for synonym in synonym_list
    ]
    # throwing an exception if any of the tokens provided in synonyms
    # does not appear in the tokens derived from sentences
    if not any(synonym in tokens for synonym in all_synonym_tokens):
        raise ValueError(
            "Received a token in synonyms that does not appear in sentences"
        )

    tokens_list = synonyms.copy()  # copy list of synonyms into a new list
    # add extracted tokens if they are not in the list already
    for token in tokens:
        if token not in all_synonym_tokens:
            tokens_list.append((token,))
    # enumerating tokens list into dictionary, making sure synonyms get the same token
    return {
        token: i for (i, tokens) in enumerate(tokens_list) for token in tokens
    }


def create_string_distance_vocab(
    sentences: list[str],
    distance: int,
    tokenizer: TokenizeFunction = tokenize_words,
    distance_function: StringDistance | None = None,
) -> dict[str, int]:
    """Create a vocabulary dictionary which pairs tokens to integers.

    Translates tokens that are within the same string distance of one another to the same integer.
    NOTE: should not be used with the tokenize_characters tokenizer!

    Args:
    ----
        sentences: list of sentences in the form of strings
        distance: distance at which strings are assumed to be the same.
        tokenizer: Function that performs tokenization on sentences.
                                                 Defaults to tokenize_words.
        distance_function: strsimpy StringDistance class to measure
                                                    string distance. Defaults to Levenshtein().

    Returns:
    -------
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values

    """
    distance_function = (
        Levenshtein() if distance_function is None else distance_function
    )

    vocab = create_vocab(sentences, tokenizer)
    tokens = np.asarray(
        list(vocab.keys())
    )  # as numpy array so I can easily index into it

    # Tokens are in the order of the number they receive from the vocabulary.
    # This means that when the tokens are enumerated,
    # I can use the ordering to optimise things a bit.
    # Which is not a luxury, this stuff is really slow.

    for index, token in enumerate(tokens, start=1):
        close_tokens = [
            distance_function.distance(token, other_token) <= distance
            for other_token in tokens[index:]
        ]
        # reading out similar tokens
        similar = tokens[index:][close_tokens]
        for similar_token in similar:
            vocab[similar_token] = vocab[token]

    return vocab
