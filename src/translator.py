import numpy as np
from typing import Callable, Protocol
from strsimpy.string_distance import StringDistance
from strsimpy.levenshtein import Levenshtein

class IllegalArgumentError(ValueError):
    pass

tokenize_function = Callable[[str], list[str]]

# * Creating basic tokenizers
def tokenize_on_spaces(sentence: str) -> list[str]:
    """Separates out tokens in the sentence based on whitespaces

    Args:
        sentence (str): string to be separated into tokens

    Returns:
        list[str]: list of string tokens
    """
    return sentence.split()

def tokenize_characters(sentence: str) -> list[str]:
    """Separates each characters in the sentence as a separate token

    Args:
        sentence (str): string to be separated into tokens

    Returns:
        list[str]: list of string tokens
    """
    return list(sentence)

class Translator:
    """Base class for Translators
    """
    def __init__(self, tokenizer: tokenize_function, vocab: dict[str, int]):
        """Initialise a Translator

        Args:
            tokenizer: a function that takes as argument a str and returns a list[str]
            vocab: a dictionary with str as keys and int as values. Note that the lowest allowed value is 0!
        """
        self.tokenize = tokenizer
        self.vocab = vocab
        assert min(vocab.values()) == 0, "Lowest value in vocab should be 0."
        assert len(vocab) == max(vocab.values()) + 1, "The largest value in the vocab should be equal to the length of the vocab."


    def encode(self, sentence: str) -> list[int]:
        return [self.vocab[token] for token in self.tokenize(sentence)]
    
    def __len__(self):
        return max(self.vocab.values()) + 1 # Adding one since arrays start at 0
    

# * Default Translator factory

def create_default_translator(sentences: list[str], tokenizer: tokenize_function=tokenize_on_spaces) -> Translator:
    """Creates a default translator from a list of sentences and a tokenizer function

    Args:
        sentences (list[str]): list of sentences to be translated.
        tokenizer (tokenize_function, optional): function with which to perform tokenization. Defaults to tokenize_on_spaces.

    Returns:
        Translator: object to perform translation from sentences to numericalised lists.
    """
    vocab = create_vocab(sentences, tokenizer)
    return Translator(tokenizer, vocab)

# * Functions for creating vocabs

def _tokenize_sentences(sentences: list[str], tokenizer: tokenize_function) -> set[str]:
    """Helper function that converts sentences into a set of unique tokens

    Args:
        sentences (list[str]): sentences to be tokenized
        tokenizer (tokenize_function): tokenizer function

    Returns:
        set[str]: set of unique tokens extracted from the sentences
    """
    sentences = list(set(sentences))  # making sure each sentence is unique, so we don't do redundant operations
    # tokenizing each sentence and then flattening to a set of tokens
    tokenized_sentences = [tokenizer(sentence) for sentence in sentences]
    tokens = {token for sentence in tokenized_sentences for token in sentence}
    return tokens


def create_vocab(sentences: list[str], tokenizer: tokenize_function) -> dict[str, int]:
    """Creates a vocabulary dictionary which pairs each unique token in the sentences with a unique integer.

    Args:
        sentences (list[str]): list of sentences in the form of strings
        tokenizer (TokenizeFunction): function to perform tokenization on the sentences.

    Returns:
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values
    """
    tokens = _tokenize_sentences(sentences, tokenizer)  # extracting unique tokens from sentences
    vocab = {token: i for (i, token) in enumerate(tokens)} # enumerating these into a dictionary
    return vocab


def create_synonym_vocab(sentences: list[str], synonyms: list[tuple[str]], tokenizer: tokenize_function) -> dict[str, int]:
    """Creates a vocabulary dictionary where synymous tokens receive the same value in the vocab.

    Args:
        sentences (list[str]): _description_
        synonyms (list[tuple]): _description_
        tokenizer (tokenize_function): _description_

    Raises:
        ValueError: _description_

    Returns:
        dict[str, int]: _description_
    """
    tokens = _tokenize_sentences(sentences, tokenizer) # extracting unique tokens from sentences
    # flatting the list of synonym tuples so it's easier to check if tokens appear in them
    all_synonym_tokens = [synonym for synonym_list in synonyms for synonym in synonym_list]
    # throwing an exception if any of the tokens provided in synonyms does not appear in the tokens derived from sentences
    if not any([synonym in tokens for synonym in all_synonym_tokens]):
        raise ValueError("Received a token in synonyms that does not appear in sentences")
    
    tokens_list = synonyms.copy() # copy list of synonyms into a new list
    [tokens_list.append((token, )) for token in tokens if token not in all_synonym_tokens] # add extracted tokens if they are not in the list already
    vocab = {token: i for (i, tokens) in enumerate(tokens_list) for token in tokens} # enumerating tokens list into dictionary, making sure synonyms get the same token
    return vocab


def create_string_distance_vocab(
        sentences: list[str],
        distance: int, 
        tokenizer: tokenize_function=tokenize_on_spaces, 
        string_distance: StringDistance=Levenshtein(),
        ):
    vocab = create_vocab(sentences, tokenizer)
    tokens = np.asarray(list(vocab.keys())) # tokens to np array for direct indexing

    # Tokens are in the order of the number they receive from the vocabulary.
    # This means that when the tokens are enumerated, I can use the ordering to optimise things a bit.
    # Which is not a luxury, this stuff is really slow. TODO fix these algorithms myself?

    for index, key in enumerate(tokens):
        # The number is assumed to be the same as the index. If it is not, we have previously changed this number and can skip this key.
        if vocab[key] < index:
            continue
        # I only need to look from this token (given by index) onwards, not backwards, which speeds things up substantially
        close_tokens = [string_distance.distance(key, token) < distance for token in tokens[index:]]
        # reading out the similar tokens
        similar = tokens[index:][close_tokens]
        if len(similar) < 1: # skipping if there were no similar tokens
            continue
        for similar_token in similar:
            # assigns the index (which is the lowest number) to all these tokens
            vocab[similar_token] = index
    return vocab

# TODO create_string_distance_vocab

# TODO eigen Levenshtein en Damerau-Levenshtein algoritmes implementeren want strsimpy is fucking traag

# TODO merge_vocabs() ? (instead of having to create tons of convenience functions?)