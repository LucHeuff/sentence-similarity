class SentenceSimilarityError(Exception):
    """Base exception class."""


class DuplicateSentencesError(SentenceSimilarityError):
    """Raised when some of the sentences passed into sentence_similarity() are duplicated."""  # noqa: E501


class InvalidWeightBaseError(SentenceSimilarityError):
    """Raised when an invalid base for the weight matrix is provided."""
