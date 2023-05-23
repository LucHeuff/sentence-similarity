import string
from pytest import raises
from hypothesis import given, assume
from hypothesis.strategies import composite, lists, text, sampled_from, booleans, integers, permutations

from src.vocab import Vocab, IllegalArgumentError, tokenizer_options, space_token

tokenizer_methods = list(tokenizer_options.keys())

# allowing everything but spaces since those are a token splitting condition
# Also removing \\ as these are escape characters and just annoying
punctuation = string.punctuation.replace('\\', '') 
alphabet = string.ascii_letters + string.digits + punctuation

@composite
def word(draw) -> str:
    word = draw(text(alphabet, min_size=1, max_size=10))
    return word

@composite
def sentence(draw) -> str:
    words = draw(lists(word(), min_size=1)) # not allowed to be empty, I'm just assuming sentences are never empty
    return " ".join(words)

# * Testing transformations from and to sentences
@given(sentence=sentence(), method=sampled_from(tokenizer_methods))
def test_transformations(sentence, method):
    split, join = tokenizer_options[method]
    assert sentence == join(split(sentence))

# * Testing Vocab

# Testing whether the correct exception is raised when an invalid method is supplied
@given(
        sentences=lists(sentence()), 
        method=text().filter(lambda s: s not in tokenizer_methods) # however unlikely, I don't want to randomly generate a valid method
        )
def test_raises_exception(sentences, method):
    with raises(IllegalArgumentError):
        Vocab(sentences, tokenize_method=method)

# * strategies for testing the vocab
@given(
        sentences = lists(sentence(), min_size=1),
        method = sampled_from(tokenizer_methods),
        lower = booleans()
)
def test_vocab(sentences: list[str], method: str, lower: bool):
    vocab = Vocab(sentences, method, lower)
    # Checking for each sentence if the vocab works as intended
    for sentence in sentences:
        sentence = sentence.lower() if lower else sentence
        assert vocab.decode(vocab.encode(sentence)) == sentence
        assert all(isinstance(item, int) for item in vocab.encode(sentence))
