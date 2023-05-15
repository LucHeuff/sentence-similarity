import pandas as pd
from pandera import SeriesSchema
from collections import Counter
from functools import partial
from typing import Callable

UNK = "<unk>" # unknown token

class IllegalArgumentError(ValueError):
    pass

split_function = Callable[[str], list[str]]
join_function = Callable[[list[str]], str]


class Tokenizer:
    def __init__(self, 
                 lower: bool, 
                 split_function: split_function, 
                 join_function: join_function
                 ) -> None:
        self.lower = lower
        self.split_function = split_function
        self.join_function = join_function

    def tokenize(self, sentence: str) -> list[str]:
        sentence = sentence.lower() if self.lower else sentence
        return self.split_function(sentence)

    def encode(self, sentence: str, encoder: dict) -> list[int]:
        return [encoder[word] for word in self.tokenize(sentence)]
    
    def decode(self, numbered_sentence: list[int], decoder: dict) -> str:
        tokens = [decoder[num] for num in numbered_sentence]
        return self.join_function(tokens)

def split_words(sentence: str) -> list[str]:
    return sentence.split()

def join_words(tokens: list[str]) -> str:
    return " ".join(tokens)

def split_characters(sentence: str) -> list[str]:
    return list(sentence)

def join_characters(tokens: list[str]) -> str:
    return "".join(tokens)

def word_tokenizer_factory(lower: bool) -> Tokenizer:
    return Tokenizer(lower, split_function=split_words, join_function=join_words)

def character_tokenizer_factory(lower: bool) -> Tokenizer:
    return Tokenizer(lower, split_function=split_characters, join_function=join_characters)

        
tokenizer_options = dict(
    words=word_tokenizer_factory,
    characters=character_tokenizer_factory
)

combine_sentences = Callable[[list[str]], str] # used to throw all the sentences together

sentence_combiner = dict(
    words=join_words,
    characters=join_characters
)

class Vocab:
    def __init__(
            self, 
            sentences: list[str], 
            tokenize_method: str='words', 
            lower: bool=False
        ) -> None:
        self.tokenizer, self.combine_sentences = self._create_tokenizer_and_combiner(tokenize_method, lower)
        self.vocab = self._create_vocab(sentences)
        self.decoder = self._create_decoder()
    
    def _create_tokenizer_and_combiner(self, tokenize_method: str, lower: bool) -> tuple[Tokenizer, combine_sentences]:
        try:
            return tokenizer_options[tokenize_method](lower), sentence_combiner[tokenize_method]
        except KeyError:
            allowed = [f"'{key}'" for key in tokenizer_options.keys()]
            raise IllegalArgumentError(f"Received an unknown tokenization method. Allowed methods are [{', '.join(allowed)}], but received '{tokenize_method}'")

    def _create_vocab(self, sentences: list[str]) -> dict[str, int]:
        sentences = list(set(sentences)) # only interested in unique sentences
        all_sentences = self.combine_sentences(sentences)
        tokens = self.tokenizer.tokenize(all_sentences)
        counter = Counter(tokens)
        vocab = {token: i for (i, token) in enumerate(counter, start=1)}
        vocab[UNK] = 0
        return vocab
    
    def _create_decoder(self) -> dict:
        return {value: key for key, value in self.vocab.items()}       

    def encode(self, sentence: str) -> list[int]:
        return self.tokenizer.encode(sentence, self.vocab)
    
    def decode(self, numbered_sentence: list[int]) -> str:
        return self.tokenizer.decode(numbered_sentence, self.decoder)
    
    def __len__(self) -> int:
        return len(self.vocab)
