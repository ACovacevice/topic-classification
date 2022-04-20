import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from text_processing import getKeywords
from tqdm import tqdm


class LDATransformer(TransformerMixin):

    """
    Transformer object for Latent Dirichlet Allocation (LDA) modeling.
    """

    def __init__(self, no_below=50, no_above=0.5, keep_n=100000):
        """
        Args:
            no_below (int, optional):
                ignore vocabs with frequency below this value. Default is 50.
            no_above (float, optional):
                ignore vocabs with percent frequency above this value. Default is 0.5.
            keep_n (int, optional):
                max number of vocabs to be stored in dictionary. Defaults to 100000.
        """
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n

    def fit(self, X, y=None, **fit_params):
        """
        Fit this object to `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Iterable of textual data.
            y (optional):
                Not implemented. Defaults to None.
        Returns:
            object: this fitted object.
        """

        if hasattr(X, "__len__"):
            size = len(X)
        else:
            size = sum(1 for x in X)

        self.data_words, self.corpus = [], []

        print("Extracting keywords...")
        pbar = tqdm(total=size)
        for x in X:
            self.data_words.append(getKeywords(x))
            pbar.update(1)

        self.dictionary = Dictionary(self.data_words)
        self.dictionary.filter_extremes(self.no_below, self.no_above, self.keep_n)

        print("Updating bag of words...")
        pbar = tqdm(total=size)
        for doc in self.data_words:
            self.corpus.append(self.dictionary.doc2bow(doc))
            pbar.update(1)

        return self

    def transform(self, X, y=None):
        """
        Apply transformation to `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Iterable of textual data.
            y (optional):
                Not implemented. Defaults to None.
        Returns:
            {tuple(data_words, corpus, dictionary)}: Transformed `X`.
        """
        data_words = [getKeywords(x) for x in X]
        corpus = [self.dictionary.doc2bow(doc) for doc in data_words]
        return data_words, corpus, self.dictionary

    def fit_transform(self, X, y=None, **fit_params):
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
        return super().fit_transform(X, **fit_params)


class LDATopicModel(TransformerMixin):

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
        """
        Args:
            num_topics (int, optional):
                The number of requested latent topics to be extracted
                from the training corpus. Defaults to 100.
            decay (float, optional):
                A number between (0.5, 1] to weight what percentage of
                the previous lambda value is forgotten when each new
                document is examined. Defaults to 0.5.
            offset (int, optional):
                Hyper-parameter that controls how much we will slow down
                the first steps the first few iterations. Defaults to 64.
            chunksize (int, optional):
                Number of documents to be used in each training chunk.
                Defaults to 4096.
            eta ({float, numpy.ndarray of float, list of float, str}, optional):
                A-priori belief on topic-word distribution, this can be:
                    scalar for a symmetric prior over topic-word distribution,
                    1D array of length equal to num_words to denote an asymmetric
                    user defined prior for each word, matrix of shape
                    (num_topics, num_words) to assign a probability for each
                    word-topic combination.
                Alternatively default prior selecting strategies can be employed
                by supplying a string:
                    ’symmetric’: Uses a fixed symmetric prior of 1.0 / num_topics,
                    ’auto’: Learns an asymmetric prior from the corpus.
                Defaults to ’symmetric’.
            passes (int, optional):
                Number of passes through the corpus during training. Defaults to 2.
            minimum_phi_value (float, optional):
                This represents a lower bound on the term probabilities. Defaults to 0.02.
            eval_every (int, optional):
                Log perplexity is estimated every that many updates. Setting this to one
                slows down training by ~2x. Defaults to 500.
            workers (int, optional):
                Number of workers processes to be used for parallelization.
                If None all available cores will be used. Defaults to None.
            random_state (int, optional):
                Seed for pseudo-random number generation. Defaults to None.
        """
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

    def fit(self, X, y=None, **fit_params):
        """
        Fit this object.
        Args:
            X ({tuple(data_words, corpus, dictionary)}):
                The vocabs, their numerical mappings and dictionary to be
                used as features.
            y (optional):
                Not implemented. Defaults to None.
        Returns:
            object: This fitted object.
        """
        data_words, corpus, dictionary = X
        self.LDA = LdaMulticore(
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
        self._coherence = self.score(X)
        return self

    def transform(self, X):
        """
        Apply transformation to `X`.
        Args:
            X ({tuple(data_words, corpus, dictionary)}):
                The vocabs, their numerical mappings and dictionary to be
                used as features.
            y (optional):
                Not implemented. Defaults to None.
        Returns:
            array: Transformed `X`.
        """
        _, corpus, _ = X

        def func():
            for doc in corpus:
                topics = self.LDA.get_document_topics(doc, per_word_topics=False)
                pred = [(g + 1, p) for g, p in topics]
                yield sorted(pred, key=lambda x: -x[1])

        return pd.Series(list(func()), name="Group")

    def predict(self, X):
        """
        Make a prediction about `X`.
        Args:
            X ({tuple(data_words, corpus, dictionary)}):
                The vocabs, their numerical mappings and dictionary to be
                used as features.
            y (optional):
                Not implemented. Defaults to None.
        Returns:
            array: Transformed `X`.
        """
        return self.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit this object, then apply transformation to `X`.
        Args:
            X ({tuple(data_words, corpus, dictionary)}):
                The vocabs, their numerical mappings and dictionary to be
                used as features.
            y (optional):
                Not implemented. Defaults to None.
        Returns:
            array: Transformed `X`.
        """
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
            model=self.LDA,
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
        if hasattr(self, "_coherence"):
            return self._coherence
        raise ValueError("Not fit yet.")


class LDAClassificationModel(ClassifierMixin, BaseEstimator):

    """
    Classification model based on Latent Dirichlet Allocation (LDA) topic modeling.
    """

    def fit(self, X, y, **fit_params):

        """
        Fit this object.
        Args:
            X (pd.Series):
                Array of lists containing tuples (LDA_topic, probability).
                Example: [[(0, 0.5), (1, 0.5)], [(0, 0.75), (1, 0.25)]].
            y (pd.Series):
                Array of target labels.
        Returns:
            object: This fitted object.
        """

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

        self.map = result

        for name in self.map:
            if name.endswith("|Group)"):
                self.p_name = name
            elif name != "Group":
                self.y_name = name

        return self

    def predict(self, X, k=-1):

        """
        Make class predictions about `X`.
        Args:
            X (array-like or iterable with shape (n_samples, 0)):
                Array of lists containing tuples (LDA_topic, probability).
                Example: [[(0, 0.5), (1, 0.5)], [(0, 0.75), (1, 0.25)]].
            k (int, optional):
                Number of potential prediction values.
                Defaults to -1, i.e., all candidates will be presented.
        Returns:
            list: prediction classes for `X`.
        """

        if not hasattr(self, "p_name"):
            raise ValueError("Not fit yet.")

        if not hasattr(self, "y_name"):
            raise ValueError("Not fit yet.")

        def retrieve(x):
            new_x = sorted(x, key=lambda x: -x[1])
            return new_x[: k if k > 0 else len(x)]

        def unstack(X):
            new_X = pd.Series(X).explode().apply(pd.Series)
            new_X.columns = ["Group", "P(Group)"]
            new_X.reset_index(drop=False, inplace=True)
            return new_X

        merged = pd.merge(unstack(X), self.map, on="Group", how="inner")
        merged["Prob"] = merged["P(Group)"] * merged[self.p_name]

        result = merged.groupby(["index", self.y_name]).agg(Prob=("Prob", "sum"))
        result.reset_index(drop=False, inplace=True)

        result["Pred"] = result[[self.y_name, "Prob"]].apply(
            lambda x: (x[0], x[1]), axis=1
        )
        result = result.groupby("index").agg(**{self.y_name: ("Pred", retrieve)})
        result.index.name = None

        return result[self.y_name]
