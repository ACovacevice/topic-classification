import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassificationModel(ClassifierMixin, BaseEstimator):

    """Classification model that maps latent topics to actual labels."""

    def __repr__(self):
        return "<ClassificationModel()>"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        dummy = [np.where(np.array(y) == cls, 1, 0) for cls in self.classes_]
        probs = np.matmul(dummy, X)
        self.T_prob_map_ = (probs / probs.sum(axis=0)).T
        return self

    def predict_proba(self, X):
        return np.matmul(X, self.T_prob_map_)

    def predict(self, X, k=1):
        probs = self.predict_proba(X)
        best_k = np.argsort(probs, axis=1)[:, -k:]
        return np.flip(self.classes_[best_k], axis=1)
