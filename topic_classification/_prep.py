import re
import string
from typing import List

from sklearn.preprocessing import FunctionTransformer


def get_unigrams(corpus: List[str]) -> List[List[str]]:
    """
    Extract unigrams (sizes 2+) from `corpus`.
    Args:
        corpus (List[str]): corpus used to fit the LDA model.
    Returns:
        List[List[str]]: token lists for each document in the corpus.
    """
    res, punct_dict = [], str.maketrans("", "", string.punctuation)
    for doc in corpus:
        tokens = doc.lower().translate(punct_dict).split()
        res.append([token for token in tokens if len(token) > 2])
    return res


class DocTransformer(FunctionTransformer):

    """Document preprocessor as a transformer object."""

    def __init__(self, func=None):
        """Constructor method for the DocTransformer class object.
        Args:
            func (optional):
                Document preprocessing function.
                If None, a basic unigram extractor will be used.
                Defaults to None.
        """
        if func is None:
            super().__init__(func=get_unigrams)
        else:
            super().__init__(func=func)

    def __repr__(self):
        return f"<DocTransformer(func={self.func.__name__})>"

    def transform(self, X, y=None, **params):
        return super().transform(X)
