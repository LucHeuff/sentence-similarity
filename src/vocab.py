import re
import string
from collections import Counter
from typing import Callable

class IllegalArgumentError(ValueError):
    pass

split_function = Callable[[str], list[str]]
join_function = Callable[[list[str]], str]

space_token = "\x07" # deliberately using a printable character that will not collide with either spaces or punctuation

def split_words(sentence: str) -> list[str]:
    sentence = re.sub(rf"\s([{string.punctuation}]+)", rf" {space_token} \1", sentence) # making a token for a space when punctuation is preceded by it
    sentence = re.sub(rf"([{string.punctuation}]+)", r" \1", sentence) # adding a space in front of punctuation if it was not already there
    return sentence.split()

def join_words(tokens: list[str]) -> str:
    sentence = " ".join(tokens)
    sentence = re.sub(rf"\s([{string.punctuation}]+)", r"\1", sentence) # remove extra space before punctuation
    sentence = re.sub(rf"\s{space_token}", " ", sentence)
    return sentence

def split_characters(sentence: str) -> list[str]:
    return list(sentence)

def join_characters(tokens: list[str]) -> str:
    return "".join(tokens)

        
tokenizer_options = dict(
    words=(split_words, join_words),
    characters=(split_characters, join_characters)
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
        self.lower = lower

        self.split_function, self.join_function = self._get_split_and_join(tokenize_method)
        self.vocab = self._create_vocab(sentences)
        self.decoder = self._create_decoder()
    
    def _get_split_and_join(self, tokenize_method: str) -> tuple[split_function, join_function]:
        try:
            return tokenizer_options[tokenize_method]
        except KeyError:
            allowed = [f"'{key}'" for key in tokenizer_options.keys()]
            raise IllegalArgumentError(f"Received an unknown tokenization method. Allowed methods are [{', '.join(allowed)}], but received '{tokenize_method}'")

    def _create_vocab(self, sentences: list[str]) -> dict[str, int]:
        sentences = list(set(sentences)) # only interested in unique sentences
        all_sentences = " ".join(sentences)
        tokens = self.tokenize(all_sentences)
        counter = Counter(tokens)
        vocab = {token: i for (i, token) in enumerate(counter)}
        return vocab
    
    def _create_decoder(self) -> dict:
        return {value: key for key, value in self.vocab.items()}    

    def tokenize(self, sentence: str) -> list[str]:
        """Converts a sentence to separate tokens.

        Args:
            sentence (str): The sentence to be tokenized

        Returns:
            list[str]: the sentence split into tokens
        """
        sentence = sentence.lower() if self.lower else sentence
        return self.split_function(sentence)

    def encode(self, sentence: str) -> list[int]:
        f"""Encodes a sentence through the vocabulary to a list of integers.

        Args:
            sentence (str): the sentence to be encoded

        Returns:
            list[int]: the encoded sentence as a list of integers
        """
        return [self.vocab[token] for token in self.tokenize(sentence)]
    
    def decode(self, numbered_sentence: list[int]) -> str:
        f"""Decodes a list of integers through the vocabulary to a sentence.

        Args:
            numbered_sentence (list[int]): list of integers representing a stence

        Returns:
            str: the decoded sentence as a string
        """
        tokens = [self.decoder[num] for num in numbered_sentence]
        return self.join_function(tokens)
    
    def __len__(self) -> int:
        return len(self.vocab)
