import string
import hypothesis.strategies as st
from hypothesis import given, assume
from hypothesis.strategies import composite
from pytest import raises
from typing import Callable

punctuation = string.punctuation.replace("\\", "")
alphabet = string.ascii_letters + string.digits + punctuation
whitespace = string.whitespace

# * General strategies

token_strategy = Callable[[], st.SearchStrategy[str]]

@composite
def character_strategy(draw) -> str:
    """Generates a single character, including whitespace"""
    character = draw(st.text(alphabet + whitespace, min_size=1, max_size=1))
    return character

@composite
def word_strategy(draw) -> str:
    """Generates a 'word' consisting of multiple characters, excluding whitespace"""
    word = draw(st.text(alphabet, min_size=1, max_size=10))
    return word

# * Testing tokenizer functions

from src.translator import tokenize_on_spaces, tokenize_characters

@composite
def word_tokens_strategy(draw) -> tuple[list[str], str]:
    """Generates a set of words and combines those words into a sentence with a whitespace.
    Allows testing whether tokenization correctly recovers the words that went into the sentence.    
    """
    words = draw(st.lists(word_strategy(), min_size=1))
    sentence = " ".join(words)
    return words, sentence

@given(data=word_tokens_strategy())
def test_tokenize_on_spaces(data):
    """Testing whether tokenizing on spaces returns the words that went into the sentence"""
    words, sentence = data
    assert tokenize_on_spaces(sentence) == words  

@composite
def character_tokens_strategy(draw) -> tuple[list[str], str]:
    """Generates a set of characters and combines these characters into a sentence by pasting them together directly.
    Allows testing whether tokenization correctly recovers the characters that went into the sentence.
    """
    characters = draw(st.lists(character_strategy(), min_size=1))
    sentence = "".join(characters)
    return characters, sentence  

@given(data=character_tokens_strategy())
def test_tokenize_characters(data: dict):
    """Testing whether tokenizing on characters returns the charactes that went into the sentence"""
    characters, sentence = data
    assert tokenize_characters(sentence) == characters # testing whether the tokenizer returns each character

# * Testing Translator

from src.translator import tokenize_function, Translator

join_func = Callable[[list[str]], str]

tokenizer_factory = {
    # method: (tokenizer, sampler, join_function)
    'on_spaces': (tokenize_on_spaces, word_strategy, lambda x: " ".join(x)),
    'characters': (tokenize_characters, character_strategy, lambda x: "".join(x))
}

@composite
def tokenizing_method_strategy(draw) -> tuple[tokenize_function, token_strategy, join_func]:
    """Since the behaviour of tokens differs between tokenizers, creating a separate strategy to keep the required functions and strategies together."""
    tokenize_method = draw(st.sampled_from(list(tokenizer_factory.keys())))
    return tokenizer_factory[tokenize_method]  # NOTE should raise an error when a new tokenizer is added but not included in the tokenizer_factory :)

@composite
def translator_strategy(draw) -> tuple[tokenize_function, dict[str, int], list[int], str]:
    """Generating a tokenizer, vocab, encodings and sentence to test whether the translator behaves properly:
    - Generating a vocab with tokens beloning to a tokenizing strategy
    - Generating a random encoding
    - Generating a sentence from that encoding through reversing the vocab. This allows testing whether the translator encodes properly.    
    """
    tokenizer, sampler, join_function = draw(tokenizing_method_strategy())
    
    randomizer = draw(st.randoms()) # creating a randomizer to shuffle values in the vocab, so it's not always 0 through max down the vocab.

    # generating vocabulary
    _vocab_length = draw(st.integers(min_value=5, max_value=10))   # varying vocab length
    keys = draw(st.lists(sampler(), min_size=_vocab_length, max_size=_vocab_length, unique=True))  # sampling tokens of equal length to the vocab
    values = list(range(_vocab_length))   # shuffling values 
    randomizer.shuffle(values) # randomizing to make sure order doesn't influence the Translator
    vocab = {key: value for (key, value) in zip(keys, values)}  # constructing the vocab
    _reverse_vocab = {value: key for (key, value) in vocab.items()} # reversing so a sentence can be generated from encodings
    # create an encoded sentence and constructing an accompanying sentence through the vocab 
    encodings = draw(st.lists(st.sampled_from(values), min_size=1))
    sentence = join_function(_reverse_vocab[num] for num in encodings)

    return tokenizer, vocab, encodings, sentence

@given(data=translator_strategy())
def test_translator(data):
    """Testing Translator:
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
    tokenizer = st.sampled_from([tokenize_on_spaces, tokenize_characters]),
    negative_vocab = st.dictionaries(keys=word_strategy(), values=st.integers(max_value=-1), min_size=5)
)
def test_translator_negative_vocab_assertion(tokenizer, negative_vocab):
    """Testing whether Translator correctly fails assertion when negative values are supplied in the vocab"""
    with raises(AssertionError):
        Translator(tokenizer, negative_vocab)

@composite
def too_large_values_vocab(draw):
    """Generating a vocab that has larger values in it than it is long"""
    _vocab_length = draw(st.integers(min_value=1, max_value=10))
    too_large_vocab = draw(st.dictionaries(
            keys=word_strategy(), 
            values=st.integers(min_value=_vocab_length+1), 
            min_size=_vocab_length, max_size=_vocab_length))
    return too_large_vocab

@given(
    tokenizer = st.sampled_from([tokenize_on_spaces, tokenize_characters]),
    too_large_vocab = too_large_values_vocab()
)
def test_translator_too_large_value_vocab_assertion(tokenizer, too_large_vocab):
    """Testing whether Translator correctly fails assertion when vocab contains values that are larger than its length"""
    with raises(AssertionError):
        Translator(tokenizer, too_large_vocab)

from src.translator import create_vocab, create_default_translator, IllegalArgumentError

@composite
def sentence_list(draw, corpus: list[str]) -> list[str]:
    """Generates a list of tokens by sampling from a provided corpus of tokens"""
    sentence = draw(st.lists(st.sampled_from(corpus), min_size=1))
    return sentence

# * Testing create_default_translator() & create_vocab()

@composite
def default_translator_strategy(draw) -> tuple[set[str], list[str], tokenize_function, join_func]:
    """Generates a corpus of tokens, sentences drawn from that corpus, a tokenizer, and a method of joining tokens according to the tokenizer.
    """
    tokenizer, sampler, join_function = draw(tokenizing_method_strategy())

    # First generating tokens to sample sentences from, then afterward reconstruction which tokens were used in the sentences as the actual corpus.
    # This is much easier than trying to make sure the sentences use all the tokens in the candidate corpus but are still random enough
    candidate_corpus = draw(st.lists(sampler(), min_size=10))    # Generating tokens to sample from 
    sentences_lists = draw(st.lists(sentence_list(candidate_corpus), min_size=5))  # generating a list of lists of tokens as sentences
    corpus = {token for sentence in sentences_lists for token in sentence} # Deducing which unique tokens were used in the sentences by flattening the lists
    sentences = [join_function(sentence) for sentence in sentences_lists] # convertings the sentence lists into proper sentences [str]

    return corpus, sentences, tokenizer, join_function

def _test_vocab(corpus: set, vocab: dict[str, int]):
    """Helper function as these tests are repeated more often.
    - Testing whether the vocab contains alls the tokens in the corpus
    - Testing whether the length of the corpus is equal to the length of the corpus (no redundant tokens)
    - Testing whether the smallest value in the vocab is 0
    - Testing whether all the values in the vocab are integers
    - Testing whether the largest value in the vocab is less than or equal to the length of the vocab
    """
    assert corpus == set(vocab.keys())  
    assert len(corpus) == len(vocab) 
    assert min(vocab.values()) == 0  
    assert set(map(type, vocab.values())) == {int}
    assert max(vocab.values()) <= len(vocab) 

@given(data=default_translator_strategy())
def test_create_vocab(data):
    """Testing creation of vocab"""
    corpus, sentences, tokenizer, _ = data
    vocab = create_vocab(sentences, tokenizer)
    _test_vocab(corpus, vocab)


@given(data=default_translator_strategy())
def test_create_default_translator(data):
    """Testing creating of default translator:
    - Testing whether all tokens in the sentences ended up in the translator vocab
    - Testing whether the length of the translator is equal to the length of the corpus
    - Testing whether encoding then decoding a sentence returns the original sentence
    - Testing whether encoding results in lists of integers    
    """
    corpus, sentences, tokenizer, join_function = data
    translator = create_default_translator(sentences, tokenizer)

    reverse_vocab = {value: key for (key, value) in translator.vocab.items()} # reversing vocab to allow decoding

    def decode(sentence: list[int]):
        return join_function([reverse_vocab[num] for num in sentence])

    assert corpus == set(translator.vocab.keys()) # set comparison 
    assert len(translator) == len(corpus) # not entirely redundant since this also tests for ints
    for sentence in sentences:  
        encoding = translator.encode(sentence)
        assert decode(encoding) == sentence # checking if reversing encoding restores the sentence
        assert all(isinstance(item, int) for item in encoding) # checking if encodings return integers

# * Testing create_synonym_vocab

from src.translator import create_synonym_vocab

@composite
def synonym_vocab_strategy(draw) -> tuple[set[str], list[str], list[tuple[str]], tokenize_function]:
    """Generate a corpus, sentences, synonyms and a tokenizer.
    Synonyms are a single list[str] sampled from the corpus.    
    """ 
    corpus, sentences, tokenizer, _ = draw(default_translator_strategy()) # start with default strategy

    assume(len(corpus) > 2)  # filtering out corpuses that are too small since this can cause trouble when sampling synonyms
    # sampling synonyms
    sample_corpus = sorted(list(corpus))  # this needs to be sorted to help hypothesis do the sampling
    synonyms = tuple(draw(st.lists(st.sampled_from(sample_corpus), min_size=2, unique=True)))  # randomly sample tokens from corpus to be synonyms

    return corpus, sentences, [synonyms], tokenizer


@given(data=synonym_vocab_strategy())
def test_create_synonym_vocab(data):
    """Testing create_synonym_vocab().
    In addition to the standard vocab tests:
    # - Test whether all the synonyms from the single synonym list get the same value in the vocab.
    """
    corpus, sentences, synonyms, tokenizer = data
    vocab = create_synonym_vocab(sentences, synonyms, tokenizer)
    _test_vocab(corpus, vocab)  # performing basic checks for vocabs
    # checking if all synonyms in each list of synonyms have received the same value
    for syn_list in synonyms:
        assert len({vocab[syn] for syn in syn_list}) == 1 # if all values are the same, length of its set should be 1
    
@composite
def synonym_exception_strategy(draw) -> tuple[list[str], list[tuple[str]], tokenize_function]:
    """Generating a corpus of tokens, sentences consisting of tokens in the corpus, a tokenizer 
    and a set of synonyms that do not appear in the sentences,
    to test whether create_synonym_vocab() correctly throws a ValueError.
    """
    corpus, sentences, tokenizer, _ = draw(default_translator_strategy())

    synonyms = draw(st.lists(word_strategy(), min_size=2))  # generating new random word tokens that are (probably) not in the corpus
    assume(all([synonym not in corpus for synonym in synonyms])) # making sure the new tokens are actually not in the corpus

    return sentences, [synonyms], tokenizer

@given(data=synonym_exception_strategy())
def test_synonym_vocab_exception(data):
    """Testing whether create_synonym_vocab() correctly throws a value error when synonym tokens are provided that do not appear in the sentences."""
    sentences, synonyms, tokenizer = data
    with raises(ValueError):
        create_synonym_vocab(sentences, synonyms, tokenizer)