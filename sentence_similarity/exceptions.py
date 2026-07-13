class SentenceSimilarityError(Exception):
    """Base exception class."""


class DuplicateSentencesError(SentenceSimilarityError):
    """Raised when some of the sentences passed into sentence_similarity() are duplicated."""  # noqa: E501


class InvalidWeightMatrixError(SentenceSimilarityError):
    """Raised when input to generate a weight matrix is invalid."""


class VocabUniqueTokensError(SentenceSimilarityError):
    """Used when all tokens for the vocab appear only once in the corpus."""


class StringDistanceVocabError(SentenceSimilarityError):
    """Raised when the distance function is not a MetricStringDistance."""
