import re

from sklearn.preprocessing import FunctionTransformer


def get_unigrams(docs):

    """
    For each document in `docs`, list unigram tokens sized above 2.
    Args:
        docs (list):
            Array or iterable containing documents (strings).
    Returns:
        list:
            List of lists containing the tokens found.
    """

    def unigrams(doc):
        tokens = re.findall(r"([\w-]+)", doc.lower())
        return [token for token in tokens if len(token) > 2]

    result = list()
    for doc in docs:
        result.append(unigrams(doc))
    return result


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
