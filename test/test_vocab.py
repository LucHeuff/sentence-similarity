
# * Tests met hypothesis

import re
import string
from pytest import raises
from typing import Callable
from hypothesis import given
from hypothesis.strategies import SearchStrategy, composite, lists, text, sampled_from, booleans

from src.vocab import Vocab, IllegalArgumentError, UNK, tokenizer_options

tokenizer_methods = list(tokenizer_options.keys())

alphabet = string.ascii_letters + string.digits + string.punctuation # allowing everything but spaces since those are a token splitting condition

@composite
def sentence(draw: Callable[[SearchStrategy[list[str]]], str]) -> str:
    words = draw(lists(text(alphabet, min_size=1, max_size=10))) # not allowed to be empty, I'm just assuming sentences are never empty
    return " ".join(words)

@composite
def sentences(draw: Callable[[SearchStrategy[list[str]]], list[str]]) -> list[str]:
    sentences = draw(lists(sentence()))
    return [sentence for sentence in sentences if not sentence == ''] # really making sure hypothesis does not keep sneakering empty sentences past me

# Helper functions, alternative ways to tokenize sentences to compare 
def tokenize_words(sentence: str) -> list[str]:
    return re.findall(r'(\S+)', sentence)

def tokenize_characters(sentence: str) -> list[str]:
    return [c for c in sentence]

# * Testing Tokenizer

@given(
        method=sampled_from(tokenizer_methods),
        lower=booleans(),
        sentence=sentence()
        )
def test_tokenizer(method: str, lower: bool, sentence: str):
    tokenizer = tokenizer_options[method](lower)

    if lower: sentence = sentence.lower() # * bit cheating but no other realistic way to reduce sentence to lowercase
    # * reducing the sentence to words, using different implementations than the actual Tokenizer
    if method == "words": 
        tokens = tokenize_words(sentence)
    else: # Note: should run into errors if I add new methods and forget to write test cases for them
        tokens = tokenize_characters(sentence)
    # * automatically creating the encoder and decoder as these are not actually part of the test
    encoder = {word: i for i, word in enumerate(tokens, start=1)}
    decoder = {value: key for key, value in encoder.items()}
    numbered_tokens = [encoder[token] for token in tokens]

    assert tokenizer.tokenize(sentence) == tokens
    assert tokenizer.encode(sentence, encoder) == numbered_tokens
    assert tokenizer.decode(numbered_tokens, decoder) == sentence
    assert tokenizer.decode(tokenizer.encode(sentence, encoder), decoder) == sentence

# * Tests voor Vocab

@given(
        sentences=sentences(), 
        method=text().filter(lambda s: s not in tokenizer_methods) # however unlikely, I don't want to randomly generate a valid method
        )
def test_raises_exception(sentences, method):
    with raises(IllegalArgumentError):
        Vocab(sentences, tokenize_method=method)

@given(
        sentences=sentences(),
        method=sampled_from(tokenizer_methods),
        lower=booleans()
)
def test_vocab(sentences: list[str], method: str, lower: bool):
    vocab = Vocab(sentences, method, lower)
    # * making a single string from lists in a dumb way but it has to be different from how Vocab does it
    flat_sentence = ''
    for sentence in sentences:
        separator = " " if method == "words" else ""
        flat_sentence += separator+sentence

    if lower: flat_sentence = flat_sentence.lower()

    # * alternately generate tokens to test against
    if method == "words": tokens = tokenize_words(flat_sentence)
    else: tokens = tokenize_characters(flat_sentence)
    items = set(tokens)

    assert len(vocab) == len(items) + 1 # UNK is added so should be one longer
    assert set(vocab.vocab.keys()).difference(items) == set([UNK])
    for sentence in sentences:
        sentence = sentence.lower() if lower else sentence
        assert vocab.decode(vocab.encode(sentence)) == sentence
        assert all(isinstance(item, int) for item in vocab.encode(sentence))
