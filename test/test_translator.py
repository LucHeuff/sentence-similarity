import string
import hypothesis.strategies as st
from hypothesis import given
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
    character = draw(st.text(alphabet + whitespace, min_size=1, max_size=1))
    return character

@composite
def word_strategy(draw) -> str:
    word = draw(st.text(alphabet, min_size=1, max_size=10))
    return word

# * Testing tokenizer functions
# - Test whether each token that went into the sentence is properly tokenized

from src.translator import tokenize_on_spaces, tokenize_characters

@composite
def word_tokens_strategy(draw) -> tuple[list[str], str]:
    words = draw(st.lists(word_strategy(), min_size=1))
    sentence = " ".join(words)
    return words, sentence

@given(data=word_tokens_strategy())
def test_tokenize_on_spaces(data):
    words, sentence = data
    assert tokenize_on_spaces(sentence) == words  # testing whether the tokenizer returns each word

@composite
def character_tokens_strategy(draw) -> tuple[list[str], str]:
    characters = draw(st.lists(character_strategy(), min_size=1))
    sentence = "".join(characters)
    return characters, sentence  

@given(data=character_tokens_strategy())
def test_tokenize_characters(data: dict):
    characters, sentence = data
    assert tokenize_characters(sentence) == characters # testing whether the tokenizer returns each character

# * Testing BaseTranslator
#  - Test whether sentences are encoded correctly
#  - Test whether the length of the Translator returns the length of the vocab

# - Test whether AssertionErrors are correctly thrown when an invalid vocab is provided

from src.translator import _tokenize_functions, TokenizeFunction, BaseTranslator

ALLOWED_TOKENIZERS = list(_tokenize_functions.keys())

join_func = Callable[[list[str]], str]

tokenizer_factory = {
    # method: (tokenizer, sampler, join_function)
    'on_spaces': (tokenize_on_spaces, word_strategy, lambda x: " ".join(x)),
    'characters': (tokenize_characters, character_strategy, lambda x: "".join(x))
}

@composite
def tokenizing_method_strategy(draw) -> tuple[TokenizeFunction, token_strategy, join_func]:
    tokenize_method = draw(st.sampled_from(ALLOWED_TOKENIZERS))
    return tokenizer_factory[tokenize_method]

@composite
def base_translator_strategy(draw) -> tuple[TokenizeFunction, dict[str, int], list[int], str]:
    tokenizer, sampler, join_function = draw(tokenizing_method_strategy())
    
    randomizer = draw(st.randoms())

    # generating vocabulary
    _vocab_length = draw(st.integers(min_value=5, max_value=10))
    keys = draw(st.lists(sampler(), min_size=_vocab_length, max_size=_vocab_length, unique=True))
    values = list(range(_vocab_length))
    randomizer.shuffle(values) # randomizing order to make sure this doesn't influence the Translator
    vocab = {key: value for (key, value) in zip(keys, values)}
    _reverse_vocab = {value: key for (key, value) in vocab.items()} # reversing so I can generate a sentence form encodings
    # create an encoded sentence and constructing an accompanying sentence through the vocab from that
    encodings = draw(st.lists(st.sampled_from(values), min_size=1))
    sentence = join_function(_reverse_vocab[num] for num in encodings)

    return tokenizer, vocab, encodings, sentence

@given(data=base_translator_strategy())
def test_base_translator(data):
    tokenizer, vocab, encodings, sentence = data
    translator = BaseTranslator(tokenizer, vocab)

    assert translator.encode(sentence) == encodings
    assert len(translator) == len(vocab)

@given(
    tokenizer = st.sampled_from([tokenize_on_spaces, tokenize_characters]),
    negative_vocab = st.dictionaries(keys=word_strategy(), values=st.integers(max_value=-1), min_size=5)
)
def test_base_translator_negative_vocab(tokenizer, negative_vocab):
    with raises(AssertionError):
        BaseTranslator(tokenizer, negative_vocab)


@composite
def too_large_values_vocab(draw):
    """Generating a vocab that has larger values in it than it is long"""
    _vocab_length = draw(st.integers(min_value=1, max_value=10))
    too_large_vocab = draw(
        st.dictionaries(
            keys=word_strategy(), 
            values=st.integers(min_value=_vocab_length+1), 
            min_size=_vocab_length, max_size=_vocab_length))
    return too_large_vocab

@given(
    tokenizer = st.sampled_from([tokenize_on_spaces, tokenize_characters]),
    too_large_vocab = too_large_values_vocab()
)
def test_base_translator_too_large_vocab(tokenizer, too_large_vocab):
    with raises(AssertionError):
        BaseTranslator(tokenizer, too_large_vocab)

# * Testing create_vocab
# - Test whether all tokens present in sentences appear in the vocab
# - Testing whether the minimum value of the vocab is 0

# * Testing DefaultTranslator
# - Test whether decoding the encoded strings results in the original string
# - Test whether encodings result in integers
# - Test whether vocab is created correctly:
#       - Check if all the used words end up in the vocab 
# - Test whether length of DefaultTranslator returns the same length as the corpus  

# - Test if trying to create a DefaultTranslator with an unknown tokenize method results in a ValueError

from src.translator import create_vocab, create_default_translator, DefaultTranslator, IllegalArgumentError

@composite
def sentence(draw, corpus) -> str:
    sentence = draw(st.lists(st.sampled_from(corpus), min_size=1))
    return sentence

@composite
def default_translator_strategy(draw) -> tuple[set, list[str], bool, str]:
    tokenizer, sampler, join_function = draw(tokenizing_method_strategy())

    # Generating tokens to sample from 
    candidate_corpus = draw(st.lists(sampler(), min_size=5))
    sentences = draw(st.lists(sentence(candidate_corpus), min_size=5))
    # Deducing which tokens were used in the sentences
    corpus = set([token for sentence in sentences for token in sentence])
    # converting the lists of tokens into a single string
    sentences = [join_function(sentence) for sentence in sentences]

    return corpus, sentences, tokenizer, join_function

@given(data=default_translator_strategy())
def test_create_vocab(data):
    corpus, sentences, tokenizer, _ = data
    vocab = create_vocab(sentences, tokenizer)

    assert corpus == set(vocab.keys())
    assert len(corpus) == len(vocab)
    assert min(vocab.values()) == 0

@given(data=default_translator_strategy())
def test_default_translator(data):
    corpus, sentences, tokenizer, join_function = data
    translator = DefaultTranslator(sentences, tokenizer)

    reverse_vocab = {value: key for (key, value) in translator.vocab.items()} # reversing vocab to allow decoding

    def decode(sentence: list[int]):
        return join_function([reverse_vocab[num] for num in sentence])

    assert corpus == set(translator.vocab.keys()) # set comparison 
    assert len(translator) == len(corpus) # not entirely redundant since this also tests for ints
    for sentence in sentences:  
        encoding = translator.encode(sentence)
        assert decode(encoding) == sentence # checking if reversing encoding restores the sentence
        assert all(isinstance(item, int) for item in encoding) # checking if encodings return integers

@given(
    sentences = st.lists(st.lists(word_strategy(), min_size=1), min_size=1),
    method = word_strategy().filter(lambda s: s not in ALLOWED_TOKENIZERS)
)
def test_create_default_translator_raises_exception(sentences, method):
    with raises(IllegalArgumentError):
        create_default_translator(sentences, method)
