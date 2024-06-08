"""Provides easier package interface"""
from sentence_similarity.sentence_similarity import (
    TokenizeFunction,
    Translator,
    sentence_similarity,
)
from sentence_similarity.translator import (
    create_string_distance_vocab,
    create_synonym_vocab,
    create_translator,
    create_vocab,
    tokenize_characters,
    tokenize_on_spaces,
    tokenize_words,
)

