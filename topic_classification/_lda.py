import numpy as np
from gensim.models.ldamulticore import LdaMulticore
from sklearn.base import BaseEstimator, TransformerMixin


class LDATransformer(TransformerMixin, BaseEstimator):

    """Parallelized Latent Dirichlet Allocation (LDA) topic model transformer object."""

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
        dtype=np.float32,
    ):
        """
        Constructor method for the LDATransformer class object.
        Args:
            num_topics (int, optional):
                The number of requested latent topics to be extracted from
                the training corpus. Defaults to 100.
            decay (float, optional):
                A number between (0.5, 1] to weight what percentage of the
                previous lambda value is forgotten when each new document
                is examined. Defaults to 0.5.
            offset (int, optional):
                Hyper-parameter that controls how much
                we will slow down the first steps the first few iterations.
                Defaults to 64.
            chunksize (int, optional):
                Number of documents to be used in each training chunk.
                Defaults to 4096.
            eta (str, optional):
                - scalar for a symmetric prior over topic-word distribution,
                - 1D array of length equal to num_words to denote an asymmetric
                user defined prior for each word,
                - matrix of shape (num_topics, num_words) to assign a probability
                for each word-topic combination.
                Alternatively default prior selecting strategies can be employed
                by supplying a string:
                'symmetric': (default) Uses a fixed symmetric prior of 1.0 / num_topics,
                'auto': Learns an asymmetric prior from the corpus.
                Defaults to "symmetric".
            passes (int, optional):
                Number of passes through the corpus during training. Defaults to 2.
            minimum_phi_value (float, optional):
                This represents a lower bound on the term probabilities. Defaults to 0.02.
            eval_every (int, optional):
                Log perplexity is estimated every that many updates.
                Setting this to one slows down training by ~2x. Defaults to 500.
            workers (int, optional):
                Number of workers processes to be used for parallelization.
                If None, all available cores (as estimated by workers=cpu_count()-1 will be used.
                Note however that for hyper-threaded CPUs, this estimation returns a too high
                number. Set workers directly to the number of your real cores (not hyperthreads)
                minus one, for optimal performance. Defaults to None.
            random_state ({np.random.RandomState, int}, optional):
                Either a randomState object or a seed to generate one.
                Useful for reproducibility. Note that results can still
                vary due to non-determinism in OS scheduling of the worker processes.
            dtype ({numpy.float16, numpy.float32, numpy.float64}, optional):
                Data-type to use during calculations inside model.
                All inputs are also converted. Defaults to numpy.float32.
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
        self.dtype = dtype

    def __repr__(self):
        string = "<LDATransformer(num_topics={0})>"
        return string.format(self.num_topics)

    def fit(self, X, y=None):
        corpus, dictionary = X
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
            dtype=self.dtype,
        )
        return self

    def __get_probs(self, corpus):
        for doc in corpus:
            topics = self.LDA_.get_document_topics(doc, per_word_topics=False)
            yield [p for _, p in topics]

    def __get_topics(self, corpus):
        for doc in corpus:
            topics = self.LDA_.get_document_topics(doc, per_word_topics=False)
            yield [g + 1 for g, _ in sorted(topics, key=lambda x: -x[1])]

    def predict(self, X, k=1):
        corpus, _ = X
        return np.vstack(list(self.__get_topics(corpus)))[:, :k]

    def transform(self, X, **params):
        corpus, _ = X
        return np.vstack(list(self.__get_probs(corpus)))

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, **fit_params)
