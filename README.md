# Sentence Similarity

This package contains an algorithm which calculates a metric of comparability between a pair of sentences.
The package allows for many sentneces to be compared amongst each other quickly (depending on available hardware).

# Installation
The package can be installed using `pip`:
```
pip install git+https://github.com/LucHeuff/sentence-similarity.git
```
or using [`poetry`](https://python-poetry.org/):
```
poetry install git+https://github.com/LucHeuff/sentence-similarity.git
```
or by adding it to `pyproject.toml`:
```
[tool.poetry.dependencies]
sentence-similarity = { git = "https://github.com/LucHeuff/sentence-similarity.git" }
```

# How does it work?

The main function of the package is `sentence_similarity(sentences)`, which calculates a similarity score between all
possible combinations of `sentences` (which needs to be a list of strings).
The similarity score ranges from 0 when the two sentences have no tokens in common, to 1 if the sentences exactly match,
to larger than 1 if one of the sentences is a subset of another sentence, or if tokens are repeated in either of the sentences.
If sentences have some but not all tokens in common, or common tokens that are not in the same places in the sentence, the score is between 0 and 1. 

**Note** that 'similarity' should be interpreted as tokens directly matching between two sentences.
The algorithm does not take synonyms into account out of the box (though you can add them manually with `create_synonym_vocab`)
and the algorithm is not smart enough to know when sentences mean the same thing in different languages
("That car is red" and "Die auto is rood" will compare to a score of 0, since they have no words in common).

# Customization

Given that there are many ways to tokenize a sentence and to construct a vocabulary, the `sentence_similarity` function
allows you to provide your own `tokenizer`, `translator` and `vocab` objects to fine tune the similarity algorithm to your own needs.

## Tokenization

A 'token' is a any set of characters in the form of a string. A `tokenizer` is any function that splits a string into a list of strings.
The package contains three tokenizers:

- `tokenize_words`: splits the sentence on spaces, and makes sure that each punctuation mark gets a separate
  token (For example, `'What?!'` is tokenized as `['What', '?', '!']`).
- `tokenize_on_spaces`: splits the sentence on spaces, giving punctuation no special treatment (`'What the?!'` is tokenized as `['What', 'the?!']`).
- `tokenize_characters`: splits each character into a separate token. May be useful when comparing identification codes
  instead of sentences. Note that different from the other methods, whitespaces receive their own token!

If these tokenizers do not suit your needs, you can also provide your own:

```
def custom_tokenizer(sentence: str) -> list[str]:
    # your code goes here
    return tokens
```
and pass it into the main function using `sentence_similarity(sentences, tokenizer=custom_tokenizer)`.

By default, `sentence_similarity(sentences)` will use the `tokenize_words` tokenizer.

## Translator

The Translator is a simple class that performs the translation by first tokenizing the sentence and
then passing it through a translation vocabulary. On initialisation, the Translator will also perform
some checks on the vocabulary that are intended to optimise performance.
For instance, the smallest value in the vocab **must be 0**, and the largest value of the vocab **must be equal to its length**.
The sentence similarity algorithm uses matrices to calculate the similarity score, and these requirements makes
sure that these matrices do not get any larger than they need to be.

`sentence_similarity(sentences)` will create a default Translator for you if you do not provide one.
It will split the sentences using the provided tokenizer (`tokenize_words` if you don't provide any), and
will create a vocabulary from all those tokens.

If you want to use a different vocabulary, you can use the `create_translator(tokenizer, vocab)` convenience function
to create a custom Translator and pass it into the main function using `sentence_similarity(sentences, translator=custom_translator)`.

Depending on the vocabulary generation method, the creation of the Translator can be very timeconsuming. In this case it
may be worthwhile to create the Translator separately, then pass it into the `sentence_similarity` function whenever it is called.
This can be especially helpful if the `sentence_similarity` function is called many times, using the same Translator.

## Vocabularies

A vocabulary is simply a dictionary which translates strings into an integers: `dict[str, int]`. Any such dictionary
can be used, so you can provide your own, as long as the **smallest integer is 0** and **the largest integer is equal to the lenght of the dictionary** (see above),

The package contains three methods of creating vocabularies:

- `create_vocab(sentences, tokenizer)`:   
  splits all sentences into tokens based on a provided `tokenizer` and gives each token a unique integer value.
- `create_synonym_vocab(sentences, synonyms, tokenizer)`:   
  allows you to additionally pass in a list of tuples, where each tuple contains all the words that are synonyms of each other
  (e.g. `[('motor', 'engine'), ('car', 'van', 'SUV'), ...]`)
  Each token in a set of synonyms is translated to the same integer value.
- `create_string_distance_vocab(sentences, distance, tokenizer, distance_function)`:   
  allows for tokens that are within `distance` from each other based on some `distance_function` to be translated to the same integer value.
  The `distance_function` is assumed to be a `StringDistance` function from the [`strsimpy`](https://github.com/luozhouyang/python-string-similarity) package (defaults to `Levenshtein`).
  This can be useful when there may be typo's in your sentences.
  **Note** that string distances can take a while to calculate!

## Weight matrix

The algorithm uses a 'weight matrix' in order to discount tokens that match but are not in the same position in the sentence.
By default, the weight matrix ranges from 1 on the diagonal to 0.1 on the bottom-left and top-right corners, decreasing linearly
along the way. The minimum value at the edges can be customised by setting the `weight_matrix_min` value to a float between 0 and 1, for example:
`sentence_similarity(sentences, weight_matrix_min=0.5)`. Setting this value will change how the score responds to tokens that match
between sentences but are in different places. Setting the value to 1. disables discounting entirely.
