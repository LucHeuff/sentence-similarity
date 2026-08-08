import string

import hypothesis.strategies as st
from hypothesis.strategies import DrawFn, composite

# --- Nomenclature
# A generator is a sampler that is intended to be reused in multiple places,
# but not directly used as a strategy for a test.

PUNCTUATION = r"%&'*+,-./:;=?@^`|~"
ALPHABET = string.ascii_letters + string.digits
WHITESPACE = string.whitespace


@composite
def character(draw: DrawFn) -> str:
    """Generate a single character, including whitespace."""
    return draw(st.text(ALPHABET + WHITESPACE + PUNCTUATION, min_size=1, max_size=1))


@composite
def word(draw: DrawFn, min_size: int = 1) -> str:
    """Generate a 'word', consisting of multiple letters or numbers."""
    return draw(st.text(ALPHABET, min_size=min_size, max_size=10))


@composite
def punctuation(draw: DrawFn) -> str:
    """Generate a single character of punctuation."""
    return draw(st.text(PUNCTUATION, min_size=1, max_size=1))


@composite
def sentence(draw: DrawFn, min_size: int = 1, max_size: int = 10) -> str:
    """Generate a single sentence, without punctuation."""
    return " ".join(
        draw(st.lists(word(), min_size=min_size, max_size=max_size, unique=True))
    )
