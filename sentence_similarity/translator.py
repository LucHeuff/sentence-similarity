import re
from typing import Callable, Sequence  # noqa: UP035

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

        """  # noqa: E501
        # validation checks
        assert vocab, "There are no tokens in the vocab."
        assert min(vocab.values()) == 0, "Smallest integer in vocab must be 0."
        vocab_max = max(vocab.values())
        assert sum(set(vocab.values())) == vocab_max * (vocab_max + 1) / 2, (
            "Integer values are not consecutive."
        )
        self.vocab = vocab

    def __getitem__(self, token: str) -> int | None:
        """Get the value for this token."""
        return self.vocab.get(token, None)

    def __contains__(self, token: str) -> bool:
        """Return whether vocab contains this token."""
        return token in self.vocab

    def __len__(self) -> int:
        """Get the length of the vocab."""
        return max(self.vocab.values()) + 1

    def __repr__(self) -> str:
        """Vocab in a readable form."""
        return (
            "Vocab(\n "
            + "\n ".join([f"{k}: {v}" for k, v in self.vocab.items()])
            + "\n)"
        )


class Translator:
    """Base class for Translators."""

    def __init__(self, tokenizer: TokenizeFunction, vocab: Vocab) -> None:
        """Initialise a Translator.

        Args:
        ----
            tokenizer: a function that takes as argument a str and returns a list[str]
            vocab: a vocabulary that translates unique tokens into integers
                   Note that the lowest allowed value is 0, and integer values must be consecutive!

        """  # noqa: E501
        self.tokenize = tokenizer
        self.vocab = vocab

    def encode(self, sentence: str) -> list[int | None]:
        """Encode each word in the sentence if it appears in the vocab, resulting in list of integers."""  # noqa: E501
        return [self.vocab[token] for token in self.tokenize(sentence)]

    def __len__(self) -> int:
        """Return the length of the Translator."""
        return len(self.vocab)

    def __repr__(self) -> str:
        """Print vocab for this translator."""
        return repr(self.vocab)


#  ---- Translator factories


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


def _default_vocab(
    sentences: list[str], tokenizer: TokenizeFunction
) -> dict[str, int]:
    tokens = {
        token
        for sentence in sentences
        for token in _tokenize_sentence(sentence, tokenizer)
    }
    return {token: i for (i, token) in enumerate(tokens)}


def create_default_vocab(sentences: list[str], tokenizer: TokenizeFunction) -> Vocab:
    """Create a vocabulary dictionary pairing unique tokens with unique integers.

    Args:
    ----
        sentences: list of sentences in the form of strings
        tokenizer: function that performs tokenization on sentences.

    Returns:
    -------
        Vocab from sentences

    """
    return Vocab(_default_vocab(sentences, tokenizer))


def create_synonym_vocab(
    sentences: list[str],
    synonyms: Sequence[tuple[str, ...]],
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
    vocab = _default_vocab(sentences, tokenizer)

    # keeping track of all synonyms that exist
    all_synonyms = {syn for syn_set in synonyms for syn in syn_set}

    # creating a list of lists in which tokens that get the same value
    # go in the same list
    tokens_list = []
    for syn_set in synonyms:
        # avoid appending an empty list if none of the synonyms in the set appear
        # in the vocab
        if all(syn not in vocab for syn in syn_set):
            continue
        tokens_list.append([syn for syn in syn_set if syn in vocab])

    for token in vocab:
        if token in all_synonyms:
            continue
        tokens_list.append([token])

    synonym_vocab = Vocab(
        {token: i for (i, tokens) in enumerate(tokens_list) for token in tokens}
    )

    assert len(synonym_vocab.vocab) == len(vocab), (
        "Vocabs should have the same number of tokens"
    )

    return synonym_vocab
