"""Contains tests for translator.py."""
import string
from itertools import chain
from typing import Callable

import hypothesis.strategies as st
from hypothesis import assume, given
from hypothesis.strategies import DrawFn, composite
from pytest import raises

PUNCTUATION = string.punctuation.replace("[\\]", "")
ALPHABET = string.ascii_letters + string.digits
WHITESPACE = string.whitespace

# Nomenclature:
# - a generator is a sampler that is meant to be reused in multiple places,
#   but not directly as a strategy for a test.
# - a strategy is a sampler that is bespoke to a function test.
#   Strategies can output multiple objects.

Sampler = Callable[[], st.SearchStrategy[str]]


@composite
def character_generator(draw: DrawFn) -> str:
    """Generate a single character, including WHITESPACE."""
    return draw(
        st.text(ALPHABET + WHITESPACE + PUNCTUATION, min_size=1, max_size=1)
    )


@composite
def word_generator(draw: DrawFn, min_size: int = 1) -> str:
    """Generate a 'word' consisting of multiple characters, excluding WHITESPACE."""
    return draw(st.text(ALPHABET, min_size=min_size, max_size=10))


@composite
def punctuation_generator(draw: DrawFn) -> str:
    """Generate a single character of punctuation."""
    return draw(st.text(PUNCTUATION, min_size=1, max_size=1))


# ---- Testing tokenizer functions ----
from sentence_similarity.translator import (  # noqa
    tokenize_characters,
    tokenize_on_spaces,
    tokenize_words,
)


@composite
def tokenize_on_spaces_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of words and combines those words into a sentence with a WHITESPACE.

    Allows testing whether tokenization correctly recovers the words that went into the sentence.
    """
    words = draw(st.lists(word_generator(), min_size=1))
    sentence = " ".join(words)
    return words, sentence


@given(data=tokenize_on_spaces_strategy())
def test_tokenize_on_spaces(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on spaces returns the words that went into the sentence."""
    words, sentence = data
    assert tokenize_on_spaces(sentence) == words


@composite
def tokenize_character_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of characters and combines these characters into a sentence by pasting them together directly.

    Allows testing whether tokenization correctly recovers the characters that went into the sentence.
    """
    characters = draw(st.lists(character_generator(), min_size=1))
    sentence = "".join(characters)
    return characters, sentence


@given(data=tokenize_character_strategy())
def test_tokenize_characters(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on characters returns the charactes that went into the sentence."""
    characters, sentence = data
    # testing whether the tokenizer returns each character
    assert tokenize_characters(sentence) == characters


@composite
def tokenize_words_strategy(draw: DrawFn) -> tuple[list[str], str]:
    """Generate a set of words, and puts random punctuation behind each word.

    Allows testing whether tokenize_words correctly splits off punctuation
    """
    words = draw(st.lists(word_generator(), min_size=1))
    # putting punctuation after each word, easiest test
    punctuation = draw(
        st.lists(
            punctuation_generator(), min_size=len(words), max_size=len(words)
        )
    )
    # making the list of tokens as I expect them to be returned
    tokens = list(chain(*zip(words, punctuation)))
    # making a list of tokens where the punctuation is pasted right behind the word
    tokens_into_sentence = [
        "".join([word, punct]) for (word, punct) in zip(words, punctuation)
    ]
    sentence = " ".join(tokens_into_sentence)
    return tokens, sentence


@given(data=tokenize_words_strategy())
def test_tokenize_words(data: tuple[list[str], str]) -> None:
    """Test whether tokenizing on words returns the words and punctuation that went into the sentence."""
    tokens, sentence = data
    assert tokenize_words(sentence) == tokens


# ---- Testing Translator ----
from sentence_similarity.translator import TokenizeFunction, Translator  # noqa

JoinFunc = Callable[[list[str]], str]

tokenizer_factory = {
    # method format: (tokenizer, sampler, join_function)
    "on_spaces": (tokenize_on_spaces, word_generator, " ".join),
    "characters": (tokenize_characters, character_generator, "".join),
}


@composite
def tokenizer_method_generator(
    draw: DrawFn,
) -> tuple[TokenizeFunction, Sampler, JoinFunc]:
    """Create a tokenizer method.

    Since the behaviour of tokens differs between tokenizers,
    defining a separate strategy to keep the required functions and strategies together.
    """
    tokenize_method = draw(st.sampled_from(list(tokenizer_factory.keys())))
    return tokenizer_factory[tokenize_method]
    # NOTE should raise an error when a new tokenizer is added but
    # not included in the tokenizer_factory :)


@composite
def vocab_generator(draw: DrawFn, sampler: Sampler) -> dict[str, int]:
    """Generate a vocab from tokens sampled with the sampler."""
    keys = draw(
        st.lists(sampler(), min_size=5, max_size=10, unique=True)
    )  # drawing random tokens
    values = list(range(len(keys)))
    randomizer = draw(st.randoms())
    randomizer.shuffle(
        values
    )  # shuffling values to make sure the order doesn't influence the Translator
    return dict(zip(keys, values))


@composite
def translator_strategy(
    draw: DrawFn,
) -> tuple[TokenizeFunction, dict[str, int], list[int], str]:
    """Generate a tokenizer, vocab, encodings and sentence to test whether the translator behaves properly.

    - Generating a vocab with tokens beloning to a tokenizing strategy
    - Generating a random encoding
    - Generating a sentence from that encoding through reversing the vocab.
      This allows testing whether the translator encodes properly.
    """
    tokenizer, sampler, join_function = draw(tokenizer_method_generator())
    vocab = draw(vocab_generator(sampler))  # generating vocabulary
    _reverse_vocab = {
        value: key for (key, value) in vocab.items()
    }  # reversing so a sentence can be generated from encodings
    # create an encoded sentence and constructing an accompanying sentence through the vocab
    encodings = draw(
        st.lists(st.sampled_from(sorted(vocab.values())), min_size=1)
    )
    sentence = join_function(_reverse_vocab[num] for num in encodings)  # type: ignore
    return tokenizer, vocab, encodings, sentence


@given(data=translator_strategy())
def test_translator(
    data: tuple[TokenizeFunction, dict[str, int], list[int], str],
) -> None:
    """Testing Translator.

    - Testing whether sentences are correctly encoded
    - Testing whether the __len__ on the translator correctly returns the length of the vocab
    - Testing whether the __len__ on the translator is an int
    """
    tokenizer, vocab, encodings, sentence = data
    translator = Translator(tokenizer, vocab)

    assert translator.encode(sentence) == encodings
    assert len(translator) == len(vocab)
    assert isinstance(len(translator), int)


@given(
    tokenizer=st.sampled_from([tokenize_on_spaces, tokenize_characters]),
    negative_vocab=st.dictionaries(
        keys=word_generator(), values=st.integers(max_value=-1), min_size=5
    ),
)
def test_translator_negative_vocab_assertion(
    tokenizer: TokenizeFunction, negative_vocab: dict
) -> None:
    """Test whether Translator correctly fails assertion when negative values are supplied in the vocab."""
    with raises(AssertionError):
        Translator(tokenizer, negative_vocab)


@composite
def too_large_values_vocab_generator(draw: DrawFn) -> None:
    """Generate vocab that has larger values in it than it is long."""
    _vocab_length = draw(st.integers(min_value=1, max_value=10))
    return draw(
        st.dictionaries(  # type:ignore
            keys=word_generator(),
            values=st.integers(min_value=_vocab_length + 1),
            min_size=_vocab_length,
            max_size=_vocab_length,
        )
    )


@given(
    tokenizer=st.sampled_from([tokenize_on_spaces, tokenize_characters]),
    too_large_vocab=too_large_values_vocab_generator(),
)
def test_translator_too_large_value_vocab_assertion(
    tokenizer: TokenizeFunction, too_large_vocab: dict
) -> None:
    """Test whether Translator correctly fails assertion when vocab contains values that are larger than its length."""
    with raises(AssertionError):
        Translator(tokenizer, too_large_vocab)


# ---- Testing vocab and default translator creators ----
from sentence_similarity.translator import (  # noqa
    create_default_translator,
    create_vocab,
)


@composite
def corpus_and_sentences_generator(
    draw: DrawFn, sampler: Callable, join_function: JoinFunc
) -> tuple[set[str], list[str]]:
    """Generate a corpus containing tokens that are used in the sentences."""
    # It is very hard to guarantee a set of sentences that use every word in the vocab.
    # So I reverse the approach, generate sentences and retrieve the corpus from those sentences.
    candidate_corpus = draw(st.lists(sampler(), min_size=10))
    # generate 'sentences' as a list of lists of tokens, easier to grab the tokens from that
    sentences_lists = draw(
        st.lists(
            st.lists(st.sampled_from(candidate_corpus), min_size=1), min_size=5
        )
    )
    corpus = {
        token for sentence in sentences_lists for token in sentence
    }  # set avoids duplication
    sentences = [join_function(sentence) for sentence in sentences_lists]
    return corpus, sentences


@composite
def default_vocab_strategy(
    draw: DrawFn,
) -> tuple[set[str], list[str], TokenizeFunction, JoinFunc]:
    """Generate a corpus of tokens, sentences drawn from that corpus, a tokenizer, and a method of joining tokens according to the tokenizer."""
    tokenizer, sampler, join_function = draw(tokenizer_method_generator())
    corpus, sentences = draw(
        corpus_and_sentences_generator(sampler, join_function)
    )
    return corpus, sentences, tokenizer, join_function


def _test_vocab(corpus: set, vocab: dict[str, int]) -> None:
    """Test assertions for vocab.

    Helper function as these tests are repeated more often.
    - Testing whether the vocab contains alls the tokens in the corpus
    - Testing whether the length of the corpus is equal to the length
      of the corpus (no redundant tokens)
    - Testing whether the smallest value in the vocab is 0
    - Testing whether all the values in the vocab are integers
    - Testing whether the largest value in the vocab is less than or
      equal to the length of the vocab
    """
    assert corpus == set(vocab.keys())
    assert len(corpus) == len(vocab)
    assert min(vocab.values()) == 0
    assert set(map(type, vocab.values())) == {int}
    assert max(vocab.values()) <= len(vocab)


@given(data=default_vocab_strategy())
def test_create_vocab(
    data: tuple[set[str], list[str], TokenizeFunction, JoinFunc],
) -> None:
    """Test creation of vocab."""
    corpus, sentences, tokenizer, _ = data
    vocab = create_vocab(sentences, tokenizer)
    _test_vocab(corpus, vocab)


@given(data=default_vocab_strategy())
def test_create_default_translator(
    data: tuple[set[str], list[str], TokenizeFunction, JoinFunc],
) -> None:
    """Test creation of default translator.

    - Testing whether all tokens in the sentences ended up in the translator vocab
    - Testing whether the length of the translator is equal to the length of the corpus
    - Testing whether encoding then decoding a sentence returns the original sentence
    - Testing whether encoding results in lists of integers
    """
    corpus, sentences, tokenizer, join_function = data
    translator = create_default_translator(sentences, tokenizer)

    reverse_vocab = {
        value: key for (key, value) in translator.vocab.items()
    }  # reversing vocab to allow decoding

    def decode(sentence: list[int]) -> str:
        return join_function([reverse_vocab[num] for num in sentence])

    assert corpus == set(translator.vocab.keys())  # set comparison
    assert len(translator) == len(
        corpus
    )  # not entirely redundant since this also tests for ints
    for sentence in sentences:
        encoding = translator.encode(sentence)
        assert (
            decode(encoding) == sentence
        )  # checking if reversing encoding restores the sentence
        assert all(
            isinstance(item, int) for item in encoding
        )  # checking if encodings return integers


# ---- Testing create_synonym_vocab() ----
from sentence_similarity.translator import create_synonym_vocab  # noqa


@composite
def synonym_vocab_strategy(
    draw: DrawFn,
) -> tuple[set[str], list[str], list[tuple[str, ...]], TokenizeFunction]:
    """Generate a corpus, sentences, synonyms and a tokenizer.

    Synonyms are a single list[str] sampled from the corpus.
    """
    tokenizer, sampler, join_function = draw(tokenizer_method_generator())
    corpus, sentences = draw(
        corpus_and_sentences_generator(sampler, join_function)
    )
    assume(len(corpus) > 2)  # noqa
    # filtering out corpuses that are too small since this can cause trouble when sampling synonyms
    sample_corpus = sorted(corpus)
    # this needs to be sorted to help hypothesis do the sampling
    synonyms = tuple(
        draw(
            st.lists(
                st.sampled_from(sample_corpus),
                min_size=3,
                max_size=5,
                unique=True,
            )
        )
    )  # randomly sample tokens from corpus to be synonyms

    return corpus, sentences, [synonyms], tokenizer


@given(data=synonym_vocab_strategy())
def test_create_synonym_vocab(
    data: tuple[set[str], list[str], list[tuple[str, ...]], TokenizeFunction],
) -> None:
    """Test create_synonym_vocab().

    In addition to the standard vocab tests:
     - Test whether all the synonyms from the single synonym list get the same value in the vocab.
    """
    corpus, sentences, synonyms, tokenizer = data
    vocab = create_synonym_vocab(sentences, synonyms, tokenizer)  # type: ignore
    _test_vocab(corpus, vocab)  # performing basic checks for vocabs
    # checking if all synonyms in each list of synonyms have received the same value
    for syn_list in synonyms:
        assert (
            len({vocab[syn] for syn in syn_list}) == 1
        )  # if all values are the same, length of its set should be 1


@composite
def synonym_exception_strategy(
    draw: DrawFn,
) -> tuple[list[str], list[list[str]], TokenizeFunction]:
    """Generate components for synomyms that should thrown an exception.

    Generate a corpus of tokens, sentences consisting of tokens in the corpus, a tokenizer
    and a set of synonyms that do not appear in the sentences,
    to test whether create_synonym_vocab() correctly throws a ValueError.
    """
    corpus, sentences, tokenizer, _ = draw(default_vocab_strategy())

    # generating new random word tokens that are (probably) not in the corpus
    synonyms = draw(st.lists(word_generator(), min_size=2))
    # making sure the new tokens are actually not in the corpus
    assume(all(synonym not in corpus for synonym in synonyms))

    return sentences, [synonyms], tokenizer


@given(data=synonym_exception_strategy())
def test_synonym_vocab_exception(
    data: tuple[list[str], list[list[str]], TokenizeFunction],
) -> None:
    """Test whether create_synonym_vocab() correctly throws a value error when synonym tokens are provided that do not appear in the sentences."""
    sentences, synonyms, tokenizer = data
    with raises(ValueError):
        create_synonym_vocab(sentences, synonyms, tokenizer)  # type: ignore


# ---- Testing create_string_distance_vocab() ----
from sentence_similarity.translator import create_string_distance_vocab  # noqa


@composite
def string_distance_vocab_strategy(
    draw: DrawFn,
) -> tuple[set[str], list[str], TokenizeFunction, int, list[tuple]]:
    """Generate a corpus, distance, words within that distance, sentences and a tokenizer.

    Words within distance are a list of tuples with word pairs within the distance.
    """
    # Creating a candidate corpus, a string distance,
    # and extra words that get string distance applied
    tokenizer, sampler, join_function = tokenizer_factory["on_spaces"]
    candidate_corpus = draw(
        st.lists(sampler(min_size=5), min_size=10, unique=True)
    )
    distance = draw(st.integers(min_value=2, max_value=4))
    n_extra_words = draw(
        st.integers(min_value=2, max_value=max(len(candidate_corpus) // 3, 2))
    )
    words_with_distance = draw(
        st.lists(
            st.sampled_from(candidate_corpus),
            min_size=n_extra_words,
            max_size=n_extra_words,
        )
    )

    # list for storing word pairs, being the original word and the one with distance added
    word_pairs = []
    for i, word in enumerate(words_with_distance):
        # only testing insertions, as testing substitutions and deletions
        # is hard and essentially tests Levenshtein, not the vocab!
        new_word = word + "a" * distance
        word_pairs.append((word, new_word))
        # replacing the original word with the distanced word
        words_with_distance[i] = new_word

    # updating the candidate corpus and sampling sentences from it
    candidate_corpus = candidate_corpus + words_with_distance
    sentences_lists = draw(
        st.lists(
            st.lists(st.sampled_from(candidate_corpus), min_size=1), min_size=5
        )
    )
    corpus = {token for sentence in sentences_lists for token in sentence}
    sentences = [join_function(sentence) for sentence in sentences_lists]

    words_with_dist_in_corpus = [
        word for word in corpus if word in words_with_distance
    ]
    # This makes sure there are distanced words used in the sentences
    assume(len(words_with_dist_in_corpus) > 0)
    # reconstructing word pairs from what is in sentences
    word_pairs = {
        (word, other_word)
        for (word, other_word) in word_pairs
        if (word in corpus and other_word in words_with_dist_in_corpus)
    }

    return corpus, sentences, tokenizer, distance, list(word_pairs)


@given(data=string_distance_vocab_strategy())
def test_string_distance_vocab(
    data: tuple[set[str], list[str], TokenizeFunction, int, list[tuple]],
) -> None:
    """Test create_string_distance_vocab().

    In addition to the standard vocab tests:
    - Test whether word pairs with a set distance between them get the same value in the vocab
    """
    corpus, sentences, tokenizer, distance, word_pairs = data
    vocab = create_string_distance_vocab(sentences, distance, tokenizer)
    _test_vocab(corpus, vocab)
    # checking if all word pairs receive the same value from the vocab
    for word, other_word in word_pairs:
        assert vocab[word] == vocab[other_word]
