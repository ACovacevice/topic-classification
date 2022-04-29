import sys
from itertools import tee
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError, check_is_fitted
from text_processing.cleaning import clean_text
from text_processing.tagging import get_keywords
from tqdm import tqdm


class LDATransformer(TransformerMixin, BaseEstimator):

    """
    Transformer object for Latent Dirichlet Allocation (LDA) modeling.
    """

    def __init__(self, no_below=50, no_above=0.5, keep_n=100000, lang="pt"):
        """
        Args:
            no_below (int, optional):
                ignore vocabs with frequency below this value. Default is 50.
            no_above (float, optional):
                ignore vocabs with percent frequency above this value. Default is 0.5.
            keep_n (int, optional):
                max number of vocabs to be stored in dictionary. Defaults to 100000.
            lang (str, optional):
                the expected language of incoming texts. Defaults to "pt".
        """
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
        self.lang = lang

    def __repr__(self):
        return (
            "<LDATransformer(no_below={0}, no_above={1}, keep_n={2}, lang={3})>".format(
                self.no_below, self.no_above, self.keep_n, self.lang
            )
        )

    def fit(self, X, y=None, text_is_processed=False, length="auto", **params):
        """
        Fit this object to `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Iterable of textual data.
            y (optional):
                Not implemented. Defaults to None.
            text_is_processed (optional):
                If True, assumes texts have been cleaned beforehand;
                perform cleaning while fitting otherwise. Defaults to False.
            length (optional):
                If "auto", infer number of inputs (takes some time for
                long iterators). If int, assume `length` to be the size
                of the input. Defaults to "auto".
        Returns:
            object: this fitted object.
        """

        called = params.get("__outside_call__", False)

        if length == "auto":
            if hasattr(X, "__len__"):
                size = len(X)
            else:
                X, X_copy = tee(X)
                size = sum(1 for _ in X_copy)
        else:
            size = length

        self.data_words_, self.corpus_ = [], []

        print("Extracting keywords...")
        with tqdm(total=size, file=sys.stdout) as pbar:
            for x in X:
                if text_is_processed:
                    clean_x = x
                else:
                    clean_x = clean_text(x, lowercase=True, drop_accents=True)
                self.data_words_.append(
                    get_keywords(clean_x, min_size=3, lang=self.lang)
                )
                pbar.update(1)

        self.dictionary_ = Dictionary(self.data_words_)
        self.dictionary_.filter_extremes(self.no_below, self.no_above, self.keep_n)

        print("Updating bag of words...")
        with tqdm(total=size, file=sys.stdout) as pbar:
            for doc in self.data_words_:
                self.corpus_.append(self.dictionary_.doc2bow(doc))
                pbar.update(1)

        if called:
            return self.data_words_, self.corpus_, self.dictionary_

        return self

    def transform(self, X, y=None, text_is_processed=False, **params):
        """
        Apply transformation to `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Iterable of textual data.
            y (optional):
                Not implemented. Defaults to None.
            text_is_processed (optional):
                If True, assumes texts have been cleaned beforehand;
                perform cleaning while fitting otherwise. Defaults to False.
        Returns:
            {tuple(data_words, corpus, dictionary)}: Transformed `X`.
        """

        check_is_fitted(self)

        data_words = []

        for x in X:
            if text_is_processed:
                clean_x = x
            else:
                clean_x = clean_text(x, lowercase=True, drop_accents=True)
            data_words.append(get_keywords(clean_x, min_size=3, lang=self.lang))

        corpus = [self.dictionary_.doc2bow(doc) for doc in data_words]

        return data_words, corpus, self.dictionary_

    def fit_transform(self, X, y=None, **params):
        """
        Fit this object, then apply transformation to `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Iterable of textual data.
            y (optional):
                Not implemented. Defaults to None.
        Returns:
            {tuple(data_words, corpus, dictionary)}: Transformed `X`.
        """
        return self.fit(X, __outside_call__=True, **params)


class LDATopicModel(TransformerMixin, BaseEstimator):

    """
    Parallelized Latent Dirichlet Allocation (LDA) model for topic modeling.
    """

    def __init__(
        self,
        num_topics=100,
        decay=0.5,
        offset=64,
        chunksize=4096,
        eta="symmetric",
        passes=2,
        minimum_phi_value=0.02,
        eval_every=500,
        workers=None,
        random_state=None,
    ):
        self.num_topics = num_topics
        self.decay = decay
        self.offset = offset
        self.chunksize = chunksize
        self.eta = eta
        self.passes = passes
        self.minimum_phi_value = minimum_phi_value
        self.eval_every = eval_every
        self.workers = workers
        self.random_state = random_state

    def __repr__(self):
        string = "<LDATopicModel(num_topics={0}, workers={1}, random_state={2})>"
        return string.format(
            self.__dict__["num_topics"],
            self.__dict__["workers"],
            self.__dict__["random_state"],
        )

    def fit(self, X, y=None, **fit_params):
        data_words, corpus, dictionary = X
        self.LDA_ = LdaMulticore(
            num_topics=self.num_topics,
            id2word=dictionary,
            corpus=corpus,
            decay=self.decay,
            offset=self.offset,
            chunksize=self.chunksize,
            eta=self.eta,
            passes=self.passes,
            minimum_probability=0,
            minimum_phi_value=self.minimum_phi_value,
            workers=self.workers,
            eval_every=self.eval_every,
            random_state=self.random_state,
            per_word_topics=True,
        )
        self.coherence_ = self.score(X)
        return self

    def transform(self, X):
        check_is_fitted(self)
        _, corpus, _ = X

        def func():
            for doc in corpus:
                topics = self.LDA_.get_document_topics(doc, per_word_topics=False)
                pred = [(g + 1, p) for g, p in topics]
                yield sorted(pred, key=lambda x: -x[1])

        return pd.Series(list(func()), name="Group")

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, **fit_params)

    def score(self, X, window_size=110, topn=100):
        """
        Evaluate the coherence metric for this model on `X`.
        Args:
            X ({tuple(data_words, corpus, dictionary)}):
                The vocabs, their numerical mappings and dictionary to be
                used as features.
            window_size (int, optional):
                The size of the window to be used for coherence measures
                using boolean sliding window as their probability estimator.
                Defaults to 110.
            topn (int, optional):
                Integer corresponding to the number of top words to be extracted
                from each topic. Defaults to 100.
        Returns:
            float: Coherence score for this model on `X`.
        """
        data_words, corpus, dictionary = X
        return CoherenceModel(
            model=self.LDA_,
            texts=data_words,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
            window_size=window_size,
            topn=topn,
        ).get_coherence()

    def get_coherence(self):
        """
        Get the coherence metric scored by this model during training.
        Returns:
            float: Coherence score for this model.
        """
        check_is_fitted(self)
        return self.coherence_


class LDALabelModel(ClassifierMixin, BaseEstimator):

    """
    Classification component which maps latent LDA groups to actual labels.
    """

    def __repr__(self):
        return "<LDALabelModel()>"

    def fit(self, X, y, **fit_params):

        if not isinstance(X, pd.Series):
            X_copy = pd.Series(X, name="Group")
        else:
            X_copy = X.copy()

        if not isinstance(y, pd.Series):
            y_copy = pd.Series(y, name="Label")
        else:
            y_copy = y.copy()

        data = pd.concat([X_copy.explode(), y_copy], axis=1)
        data[f"P({y_copy.name}|Group)"] = data["Group"].apply(lambda x: x[1])
        data["Group"] = data["Group"].apply(lambda x: x[0])

        events = data.groupby(["Group", y_copy.name], as_index=False)
        events = events.agg(Count=(f"P({y_copy.name}|Group)", "sum"))

        total = data.groupby("Group", as_index=False)
        total = total.agg(Sum=(f"P({y_copy.name}|Group)", "sum"))

        result = pd.merge(events, total, on="Group", how="inner")
        result[f"P({y_copy.name}|Group)"] = result["Count"] / result["Sum"]
        result.drop(["Count", "Sum"], axis=1, inplace=True)

        self.map_ = result

        for name in self.map_:
            if name.endswith("|Group)"):
                self.p_name_ = name
            elif name != "Group":
                self.y_name_ = name

        return self

    def predict(self, X, k=-1):

        check_is_fitted(self)

        def retrieve(x):
            new_x = sorted(x, key=lambda x: -x[1])
            return new_x[: k if k > 0 else len(x)]

        def unstack(X):
            new_X = pd.Series(X).explode().apply(pd.Series)
            new_X.columns = ["Group", "P(Group)"]
            new_X.reset_index(drop=False, inplace=True)
            return new_X

        merged = pd.merge(unstack(X), self.map_, on="Group", how="inner")
        merged["Prob"] = merged["P(Group)"] * merged[self.p_name_]

        result = merged.groupby(["index", self.y_name_]).agg(Prob=("Prob", "sum"))
        result.reset_index(drop=False, inplace=True)

        result["Pred"] = result[[self.y_name_, "Prob"]].apply(
            lambda x: (x[0], x[1]), axis=1
        )
        result = result.groupby("index").agg(**{self.y_name_: ("Pred", retrieve)})
        result.index.name = None

        return result[self.y_name_]


class LDAClassifier(Pipeline, BaseEstimator):

    """Classification model based on Latent Dirichlet Allocation (LDA) topic modeling."""

    def __init__(
        self,
        no_below=50,
        no_above=0.5,
        keep_n=100000,
        lang="pt",
        num_topics=100,
        decay=0.5,
        offset=64,
        chunksize=4096,
        eta="symmetric",
        passes=2,
        minimum_phi_value=0.02,
        eval_every=500,
        workers=None,
        random_state=None,
    ):

        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
        self.lang = lang
        self.num_topics = num_topics
        self.decay = decay
        self.offset = offset
        self.chunksize = chunksize
        self.eta = eta
        self.passes = passes
        self.minimum_phi_value = minimum_phi_value
        self.eval_every = eval_every
        self.workers = workers
        self.random_state = random_state

        cachedir = mkdtemp()

        super().__init__(
            [
                ("transformer", LDATransformer(no_below, no_above, keep_n, lang)),
                (
                    "LDA",
                    LDATopicModel(
                        num_topics,
                        decay,
                        offset,
                        chunksize,
                        eta,
                        passes,
                        minimum_phi_value,
                        eval_every,
                        workers,
                        random_state,
                    ),
                ),
                ("classifier", LDALabelModel()),
            ],
            memory=cachedir,
        )

    def __repr__(self):
        return "<LDAClassifier(%s)>" % ", ".join(
            [
                f"{k}={v}"
                for k, v in self.__dict__.items()
                if k
                in [
                    "no_below",
                    "no_above",
                    "keep_n",
                    "num_topics",
                    "lang",
                    "workers",
                    "random_state",
                ]
            ]
        )

    def predict(self, X, k=-1):
        """
        Make a prediction about `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Texts to be processed.
            k (int, optional):
                Number of potential prediction values.
                Defaults to -1, i.e., all candidates will be presented.
        Returns:
            array: Transformed `X`.
        """
        if k > -1:
            return super().predict(X).apply(lambda x: x[:k])
        return super().predict(X)

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
        y_pred = super().predict(X).apply(lambda x: x[0][0]).values
        return f1_score(np.array(y), y_pred, average=average)

    def update_num_topics(self, num_topics):
        """
        Updates the number of topics expected from the LDATopicModel component.
        Warning: this erases all fit data from both LDATopicModel and LDALabelModel components.
        Args:
            num_topics (int):
                The new argument value for `num_topics`.
        Returns:
            None.
        """
        self.num_topics = num_topics
        self.named_steps["LDA"].num_topics = num_topics
        try:
            check_is_fitted(self)
            del self.named_steps["LDA"].LDA_
            del self.named_steps["LDA"].coherence_
            del self.named_steps["classifier"].map_
            del self.named_steps["classifier"].y_name_
            del self.named_steps["classifier"].p_name_
        except NotFittedError:
            pass
