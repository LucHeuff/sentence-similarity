import numpy as np
from typing import Callable
from strsimpy.string_distance import StringDistance
from strsimpy.levenshtein import Levenshtein

class IllegalArgumentError(ValueError):
    pass

tokenize_function = Callable[[str], list[str]]
# ---- Creating basic tokenizers ----

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

# ---- Translator ----

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
    

#  Default Translator factory
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

# ---- Functions for creating vocabs ---- 

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
        tokenizer (tokenize_function): function that performs tokenization on sentences.

    Returns:
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values
    """
    tokens = sorted(_tokenize_sentences(sentences, tokenizer))  # extracting unique tokens from sentences
    vocab = {token: i for (i, token) in enumerate(tokens)} # enumerating these into a dictionary
    return vocab


def create_synonym_vocab(sentences: list[str], synonyms: list[tuple[str]], tokenizer: tokenize_function) -> dict[str, int]:
    """Creates a vocabulary dictionary which pairs tokens to integers.
    Allows passing in lists of synonyms which translate to the same integer.

    Args:
        sentences (list[str]): list of sentences in the form strings 
        synonyms (list[tuple[str]]): list of tuples, where each tuple is filled with all the words that are synonyms of each other 
        tokenizer (tokenize_function): function that performs tokenization on sentences.

    Raises:
        ValueError: when there are tokens in a synonym tuple that do not appear in the sentences.

    Returns:
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values
    """
    tokens = sorted(_tokenize_sentences(sentences, tokenizer)) # extracting unique tokens from sentences
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
        distance_function: StringDistance=Levenshtein(),
        ) -> dict[str, int]:
    """Creates a vocabulary dictionary which pairs tokens to integers.
    Translates tokens that are within the same string distance of one another to the same integer.
    NOTE: should not be used with the tokenize_characters tokenizer!

    Args:
        sentences (list[str]): list of sentences in the form of strings
        distance (int): distance at which strings are assumed to be the same.
        tokenizer (tokenize_function, optional): Function that performs tokenization on sentences. Defaults to tokenize_on_spaces.
        distance_function (StringDistance, optional): strsimpy StringDistance class to measure string distance. Defaults to Levenshtein().

    Returns:
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values
    """
    vocab = create_vocab(sentences, tokenizer)
    tokens = np.asarray(list(vocab.keys())) # as numpy array so I can easily index into it

    # Tokens are in the order of the number they receive from the vocabulary.
    # This means that when the tokens are enumerated, I can use the ordering to optimise things a bit.
    # Which is not a luxury, this stuff is really slow. 

    for index, token in enumerate(tokens, start=1):
        close_tokens = [distance_function.distance(token, other_token) <= distance for other_token in tokens[index:]]
        # reading out similar tokens
        similar = tokens[index:][close_tokens]
        for similar_token in similar:
            vocab[similar_token] = vocab[token]
        
    return vocab
