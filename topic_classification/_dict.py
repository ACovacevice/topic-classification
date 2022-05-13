import re

from gensim.corpora import Dictionary
from sklearn.base import BaseEstimator, TransformerMixin


class DictTransformer(TransformerMixin, BaseEstimator):

    """
    Dictionary transformer class.
    """

    def __init__(self, no_below=50, no_above=0.5, keep_n=100000):
        """
        Create a new DictTransformer instance object.
        Args:
            no_below (int, optional):
                ignore tokens with frequency below this value. Default is 50.
            no_above (float, optional):
                ignore tokens with percent frequency above this value. Default is 0.5.
            keep_n (int, optional):
                max number of unique tokens to be stored in dictionary. Defaults to 100000.
        """
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n

    def __repr__(self):
        return "<DictTransformer(no_below={0}, no_above={1}, keep_n={2})>".format(
            self.no_below, self.no_above, self.keep_n
        )

    def fit(self, X, y=None, **fit_params):
        """
        Fit this object to `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Iterable of either BoW dicts or token lists.
            y (optional):
                Just a placeholder. Defaults to None.
        Returns:
            LDATransformer: this fitted object.
        """
        self.dictionary_ = Dictionary()
        self.dictionary_.add_documents(X)
        self.dictionary_.filter_extremes(self.no_below, self.no_above, self.keep_n)
        return self

    def transform(self, X, y=None, **params):
        """
        Apply transformation to `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Iterable of textual data or token lists.
            y (optional):
                Just a placeholder. Defaults to None.
        Returns:
            {tuple(corpus, dictionary)}: Transformed `X`.
        """
        corpus = []
        for doc in X:
            corpus.append(self.dictionary_.doc2bow(doc))
        return corpus, self.dictionary_
