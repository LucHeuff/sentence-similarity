import pandas as pd
from collections import Counter
from typing import Callable

class IllegalArgumentError(ValueError):
    pass

split_function = Callable[[str], list[str]]
join_function = Callable[[list[str]], str]


class Tokenizer:
    """Used to tokenize and de-tokenize string sentences.
    Not intended to be created directly, but generated through factory that fills in split and join functions.
    """
    def __init__(self, 
                 lower: bool, 
                 split_function: split_function, 
                 join_function: join_function
                 ) -> None:
        """
        Args:
            lower (bool): whether strings should be converted to lower case
            split_function: function with which to split string sentences
            join_function: function with which to recombine tokens into string sentences
        """
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
    """Generates a Tokenizer that splits sentences into words on spaces, 
    and can recombine them correctly.

    Args:
        lower (bool): whether all strings should be converted to lowercase.
                      NOTE: this means that the Tokenizer cannot restore case sensitivity when decoding!
    """
    return Tokenizer(lower, split_function=split_words, join_function=join_words)

def character_tokenizer_factory(lower: bool) -> Tokenizer:
    """Generates a Tokenizer that splits sentences into separate characters, 
    and can recombine them correctly.

    Args:
        lower (bool): whether all strings should be converted to lowercase.
                      NOTE: this means that the Tokenizer cannot restore case sensitivity when decoding!
    """
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
    """Maintains a vocabulary that can translate sentences to numbers and back."""
    def __init__(
            self, 
            sentences: list[str], 
            tokenize_method: str='words', 
            lower: bool=False
        ) -> None:
        """
        Args:
            sentences (list[str]): all sentences from which the vocabulary should be generated.
            tokenize_method (str, optional): whether tokenization should separate out 'words' or 'characters'. Defaults to 'words'.
            lower (bool, optional): whether sentences need to be converted to lowercase prior to generating the vocabulary and encoding. Defaults to False.
        """
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
        vocab = {token: i for (i, token) in enumerate(counter)}
        return vocab
    
    def _create_decoder(self) -> dict:
        return {value: key for key, value in self.vocab.items()}       

    def encode(self, sentence: str) -> list[int]:
        f"""Encodes a sentence through the vocabulary to a list of integers.

        Args:
            sentence (str): the sentence to be encoded

        Returns:
            list[int]: the encoded sentence as a list of integers
        """
        return self.tokenizer.encode(sentence, self.vocab)
    
    def decode(self, numbered_sentence: list[int]) -> str:
        f"""Decodes a list of integers through the vocabulary to a sentence.

        Args:
            numbered_sentence (list[int]): list of integers representing a stence

        Returns:
            str: the decoded sentence as a string
        """
        return self.tokenizer.decode(numbered_sentence, self.decoder)
    
    def __len__(self) -> int:
        return len(self.vocab)
