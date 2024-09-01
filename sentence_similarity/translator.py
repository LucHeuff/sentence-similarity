"""Contains Translator, tokenizer and vocabulary functions to allow customisation of sentence similarity calculations."""

import re
from collections import Counter
from typing import Callable, Iterator

from strsimpy.levenshtein import Levenshtein
from strsimpy.string_distance import MetricStringDistance

TokenizeFunction = Callable[[str], list[str]]

# ---- Creating basic tokenizers ----


def tokenize_words(sentence: str) -> list[str]:
    """Separate out tokens in sentence based on whitespaces.

    Will tokenize punctuation as a separate token, even if they are
    directly connected to a word.
    NOTE: brackets and _ are not considered punctuation.

    Args:
    ----
        sentence: string to be separated into tokens

    Returns:
    -------
        list of string tokens

    """
    punctuation = r"%&'*+,-./:;=?@\^`|~"
    sentence = re.sub(f"([{punctuation}])", "  \\1", sentence)
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


# ---- Vocab and Translator


class Vocab:
    """Base class for Vocabs."""

    vocab: dict[str, int]

    def __init__(self, vocab: dict[str, int]) -> None:
        """Initiate vocab.

        Args:
        ----
            vocab: dictionary converting tokens to integers.
                   NOTE: Smallest integer must be 0, and integer values must be consecutive
                   (but don't have to be ordered).

        """
        # validation checks
        assert vocab, "There are no tokens in the vocab."
        assert min(vocab.values()) == 0, "Smallest integer in vocab must be 0."
        vocab_max = max(vocab.values())
        assert (
            sum(set(vocab.values())) == vocab_max * (vocab_max + 1) / 2
        ), "Integer values are not consecutive."
        self.vocab = vocab

    def __getitem__(self, token: str) -> int | None:
        """Get the value for this token."""
        return self.vocab.get(token, None)

    def __iter__(self) -> Iterator[int]:
        """Get an iterator over vocab values."""
        yield from self.vocab.values()

    def __contains__(self, token: str) -> bool:
        """Return whether vocab contains this token."""
        return token in self.vocab

    def __len__(self) -> int:
        """Get the length of the vocab."""
        return max(self.vocab.values()) + 1


class Translator:
    """Base class for Translators."""

    def __init__(self, tokenizer: TokenizeFunction, vocab: Vocab) -> None:
        """Initialise a Translator.

        Args:
        ----
            tokenizer: a function that takes as argument a str and returns a list[str]
            vocab: a vocabulary that translates unique tokens into integers
                   Note that the lowest allowed value is 0, and integer values must be consecutive!

        """
        self.tokenize = tokenizer
        self.vocab = vocab

    def encode(self, sentence: str) -> list[int | None]:
        """Encode each word in the sentence if it appears in the vocab, resulting in list of integers."""
        return [self.vocab[token] for token in self.tokenize(sentence)]

    def __len__(self) -> int:
        """Return the length of the Translator."""
        return len(self.vocab)


#  ---- Translator factories


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
    vocab = create_default_vocab(sentences, tokenizer)
    return Translator(tokenizer, vocab)


def create_translator(
    vocab: dict[str, int], tokenizer: TokenizeFunction
) -> Translator:
    """Create a Translator, given a vocab and a tokenizer.

    Args:
    ----
        vocab: the desired vocabulary for the translator.
               NOTE: Smallest integer must be 0, and integer values must be consecutive
               (but don't have to be ordered).
        tokenizer: the desired tokenizer for the translator

    Returns:
    -------
        A Translator instance using the provided tokenizer and vocab

    """
    return Translator(tokenizer, create_vocab(vocab))


# ---- Vocab factories


def create_vocab(vocab: dict[str, int]) -> Vocab:
    """Create a Vocab object from a vocabulary dictionary.

    Args:
    ----
        vocab: dictonary translating tokens to integers.
               NOTE: Smallest integer must be 0, and integer values must be consecutive
               (but don't have to be ordered).

    Returns:
    -------
       Vocab constructed from provided vocabulary

    """
    return Vocab(vocab)


def _tokenize_sentence(sentence: str, tokenizer: TokenizeFunction) -> set[str]:
    """Tokenize a sentence into a unique set of tokens.

    Args:
    ----
        sentence: str containing text to be tokenized
        tokenizer: method of tokenizing

    Returns:
    -------
       set of unique tokens in this sentence

    """
    return set(tokenizer(sentence))


def _create_token_counter(
    sentences: list[str], tokenizer: TokenizeFunction
) -> Counter:
    """Create a counter, counting how often tokens appear in the corpus.

    Args:
    ----
        sentences: list of strings containing the sentences to be compared
        tokenizer: method of tokenizing

    Returns:
    -------
        Counter object with token counts.

    """
    # making sure each sentence is unique, so no redundant operations
    sentences = list(set(sentences))
    # extracting tokens from sentences. Tokens are unique in sentences,
    # but can appear multiple times between sentences
    tokens = [
        token
        for sentence in sentences
        for token in _tokenize_sentence(sentence, tokenizer)
    ]
    return Counter(tokens)


class VocabUniqueTokensError(Exception):
    """Used when all tokens for the vocab appear only once in the corpus."""


def _get_vocab_tokens(counter: Counter) -> list[str]:
    """Retrieve all tokens that occur in the vocab more than once.

    Args:
    ----
        counter: Counter object containing tokens and counts.

    Returns:
    -------
       list of tokens appearing more than once

    Raises:
    ------
        VocabUniqueTokensError: When all tokens appear only once

    """
    vocab_tokens = [token for (token, number) in counter.items() if number > 1]
    if not vocab_tokens:
        message = "All tokens appear only once in the corpus, there is no similarity to calculate."
        raise VocabUniqueTokensError(message)
    return vocab_tokens


def create_default_vocab(
    sentences: list[str], tokenizer: TokenizeFunction
) -> Vocab:
    """Create a vocabulary dictionary which pairs each unique token in the sentences with a unique integer.

    Args:
    ----
        sentences: list of sentences in the form of strings
        tokenizer: function that performs tokenization on sentences.

    Returns:
    -------
        Vocab from sentences

    """
    counter = _create_token_counter(sentences, tokenizer)
    vocab_tokens = _get_vocab_tokens(counter)
    return create_vocab({token: i for (i, token) in enumerate(vocab_tokens)})


def create_synonym_vocab(
    sentences: list[str],
    synonyms: list[tuple[str, ...]],
    tokenizer: TokenizeFunction,
) -> Vocab:
    """Create a vocabulary dictionary which pairs tokens to integers.

    Allows passing in lists of synonyms which translate to the same integer.

    Args:
    ----
        sentences: list of sentences in the form strings
        synonyms: list of tuples, where each tuple is filled with
                                     all the words that are synonyms of each other
        tokenizer: function that performs tokenization on sentences.

    Returns:
    -------
        Vocab from sentences and synonyms

    """
    counter = _create_token_counter(sentences, tokenizer)

    # update the Counter to make sure I don't remove tokens for which synonyms do appear
    for synonym_set in synonyms:
        # taking the sum of all appearences of synonyms and setting that into the counter
        token_sum = sum(counter[sym] for sym in synonym_set if sym in counter)
        for syn in synonym_set:
            counter[syn] = token_sum

    # removing tokens that only appear once
    vocab_tokens = _get_vocab_tokens(counter)

    # reducing synonyms to sets of synonyms that appear in vocab_tokens
    synonym_sets = []
    for synonym_set in synonyms:
        if all(syn not in counter for syn in synonym_set):
            continue
        synonym_appear_set = {syn for syn in synonym_set if syn in vocab_tokens}
        synonym_sets.append(synonym_appear_set)

    all_synonyms = [syn for synonym_set in synonym_sets for syn in synonym_set]

    tokens_list = synonym_sets.copy()  # copy list of synonyms into a new list
    # add extracted tokens if they are not in the list already
    for token in vocab_tokens:
        if token not in all_synonyms:
            tokens_list.append({token})

    # enumerating tokens list into dictionary, making sure synonyms get the same token
    return create_vocab(
        {token: i for (i, tokens) in enumerate(tokens_list) for token in tokens}
    )


class StringDistanceVocabError(Exception):
    """Raised when the distance function is not a MetricStringDistance."""


def create_string_distance_vocab(
    sentences: list[str],
    distance: int,
    tokenizer: TokenizeFunction = tokenize_words,
    distance_function: MetricStringDistance = Levenshtein(),  # noqa: B008
) -> Vocab:
    """Create a vocabulary dictionary which pairs tokens to integers.

    Translates tokens that are within the same string distance of one another to the same integer.
    NOTE: should not be used with the tokenize_characters tokenizer!

    Args:
    ----
        sentences: list of sentences in the form of strings
        distance: distance at which strings are assumed to be the same.
        tokenizer: Function that performs tokenization on sentences. Defaults to tokenize_words.
        distance_function: strsimpy MetricStringDistance subclass to measure string distance. Defaults to Levenshtein().

    Returns:
    -------
        dict[str, int]: vocabulary dictionary with tokens (str) keys and int values

    """
    if not isinstance(distance_function, MetricStringDistance):
        message = f"distance_function must be a MetricStringDistance, which '{type(distance_function)}' is not."
        raise StringDistanceVocabError(message)

    counter = _create_token_counter(sentences, tokenizer)

    # Calculating distance metrics if often costly, so I want to avoid repeating calculations.
    # By requiring the distance_function to be a metric, we know it is symmetric.
    # This means that if I find tokens that are similar to token a on the same distance metric,
    # I don't have to calculate the reverse again.
    # Hence making sets for seen and unseen, in an attempt to improve performance.
    unseen_tokens = set(counter)
    seen_tokens = set()
    close_sets = []
    for token in counter:
        # ignoring tokens we have already seen
        if token in seen_tokens:
            continue
        # creating a set of tokens that are within the distance threshold for this token
        close_tokens = {token} | {
            other_token
            for other_token in unseen_tokens
            if distance_function.distance(token, other_token) <= distance
        }
        # calculating how often these appear combined in the counter
        token_sum = sum(counter[close] for close in close_tokens)
        # pass on updating counter and close_sets if this set of tokens (could just be token!)
        # does not appear more than once in total in the corpus
        if token_sum > 1:
            # updating the counter for each of the close tokens
            for close in close_tokens:
                counter[close] = token_sum
            close_sets.append(close_tokens)
        # updating the seen and unseen sets
        seen_tokens = seen_tokens | set(close_tokens)
        unseen_tokens = unseen_tokens - seen_tokens

    # removing tokens that only appear once
    vocab_tokens = _get_vocab_tokens(counter)

    # copy sets of close tokens into a new list
    tokens_list = close_sets.copy()
    # add extracted tokens if they are not in the list already
    for token in vocab_tokens:
        if token not in seen_tokens:
            tokens_list.append({token})

    return create_vocab(
        {token: i for (i, tokens) in enumerate(tokens_list) for token in tokens}
    )
