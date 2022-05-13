from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from topic_classification import ClassificationModel


class LDAClassifier(Pipeline, BaseEstimator):

    """Classification pipeline based on Latent Dirichlet Allocation (LDA) topic modeling."""

    def __init__(self, doc_transformer, dict_transformer, lda_transformer):

        self.doc_transformer = doc_transformer
        self.dict_transformer = dict_transformer
        self.lda_transformer = lda_transformer

        transformer = Pipeline(
            [
                ("preprocessing", doc_transformer),
                ("dictionary", dict_transformer),
                ("LDA", lda_transformer),
            ]
        )

        super().__init__(
            [
                ("transform", transformer),
                ("classification", ClassificationModel()),
            ],
        )

    def __repr__(self):
        string = "<LDAClassifier(num_topics={0})>"
        return string.format(self.lda_transformer.num_topics)

    def get_transformer(self):
        """Get this objects's transform pipeline."""
        return self.named_steps["transform"]

    def get_classifier(self):
        """Get this objects's classification pipeline."""
        return self.named_steps["classification"]

    def predict(self, X, k=1):
        """
        Predict labels based on `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Texts to be analyzed.
            k (int, optional):
                Number of best candidates. Defaults to 1.
        Returns:
            array or arrays:
                The predicted labels.
        """
        if k > 0:
            Xt = self.get_transformer().transform(X)
            return self.get_classifier().predict(Xt, k)
        return super().predict(X)

    def predict_topic(self, X, k=1):
        """
        Predict latent topics based on `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Texts to be analyzed.
            k (int, optional):
                Number of best candidates. Defaults to 1.
        Returns:
            array or arrays:
                The predicted latent topics.
        """
        return self.get_transformer().predict(X, k=k)

    def score(self, X, y, average="weighted"):
        """
        Evaluate the F1-Score metric for this model on `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Texts to be processed.
            y (array-like or iterable with shape (n_samples, 0)):
                The true classes.
            average (str):
                The aggregation method for the F1-Score.
                Options are ["weighted", "micro", "macro"].
                Defaults to "weighted".
        Returns:
            float: F1-score metric for this model on `X`.
        """
        if average not in ["weighted", "micro", "macro"]:
            raise ValueError("`average` must be either 'micro', 'macro' or 'weighted'.")
        y_pred = super().predict(X)
        return f1_score(y, y_pred, average=average)
