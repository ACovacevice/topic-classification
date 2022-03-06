import pandas as pd
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

from typing import List

from text_processing import get_keywords_from_text

from sklearn.base import TransformerMixin, ClassifierMixin, BaseEstimator
from sklearn.pipeline import Pipeline


class LDATransformer(TransformerMixin):
    
    def __init__(self, no_below: int = 50, no_above: float = 0.5, keep_n: int = 100000):
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
    
    def fit(self, X: pd.Series, y: pd.Series = None, **fit_params) -> object:
        self.data_words = [get_keywords_from_text(x) for x in X]
        self.dictionary = Dictionary(self.data_words)
        self.dictionary.filter_extremes(self.no_below, self.no_above, self.keep_n)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.data_words]
        return self
    
    def transform(self, X: pd.Series, y: pd.Series = None) -> (List, List, Dictionary):
        data_words = [get_keywords_from_text(x) for x in X]
        corpus = [self.dictionary.doc2bow(doc) for doc in data_words]
        return data_words, corpus, self.dictionary


class LDAModel(TransformerMixin):

    def __init__(
        self, 
        num_topics: int = 100, 
        decay: float = 0.5, 
        offset: int = 64, 
        chunksize: int = 4096,
        eta: str = "symmetric",
        passes: int = 2,
        minimum_phi_value: float = 0.02,
        workers: int = 4,
        eval_every: int = 500,
        random_state: int = None,
    ):
        self.num_topics = num_topics
        self.decay = decay
        self.offset = offset
        self.chunksize = chunksize
        self.eta = eta
        self.passes = passes
        self.minimum_phi_value = minimum_phi_value
        self.workers = workers
        self.eval_every = eval_every
        self.random_state = random_state

    def fit(self, X: pd.Series, y: pd.Series = None, **fit_params) -> object:
        data_words, corpus, dictionary = X
        self.LDA = LdaMulticore(
            num_topics = self.num_topics,
            id2word = dictionary,
            corpus = corpus,
            decay = self.decay,
            offset = self.offset,
            chunksize = self.chunksize,
            eta = self.eta,
            passes = self.passes,
            minimum_probability = 0,
            minimum_phi_value = self.minimum_phi_value,
            workers = self.workers,
            eval_every = self.eval_every,
            random_state = self.random_state,
            per_word_topics = True,
        )
        self.coherence = self.score(data_words, corpus, dictionary)
        return self

    def transform(self, X: List) -> pd.Series:
        data_words, corpus, dictionary = X
        def func():
            for doc in corpus:
                topics = self.LDA.get_document_topics(doc, per_word_topics = False)
                pred = [(g + 1, p) for g, p in topics]
                yield sorted(pred, key = lambda x: -x[1])
        return pd.Series(list(func()), name="Group")
    
    def score(
        self, 
        data_words: List, 
        corpus: List, 
        dictionary: Dictionary, 
        window_size: int = 110, 
        topn: int = 100
    ) -> float:
        
        return CoherenceModel(
            model = self.LDA,
            texts = data_words,
            corpus = corpus,
            dictionary = dictionary,
            coherence = "c_v",
            window_size = window_size,
            topn = topn,
        ).get_coherence()
    
    def get_coherence(self) -> (float, str):
        if hasattr(self, "coherence"):
            return self.coherence
        raise ValueError("Not fit yet")


class LDAClassifier(ClassifierMixin, BaseEstimator):
    
    def fit(self, X: pd.Series, y: pd.Series, **fit_params) -> object:
        
        data = pd.concat([X.explode(), y], axis=1)
        data[f"P({y.name}|Group)"] = data["Group"].apply(lambda x: x[1])
        data["Group"] = data["Group"].apply(lambda x: x[0])
        
        events = data.groupby(["Group", y.name], as_index = False)
        events = events.agg(Count = (f"P({y.name}|Group)", "sum"))
        
        total = data.groupby("Group", as_index = False)
        total = total.agg(Sum = (f"P({y.name}|Group)", "sum"))
        
        result = pd.merge(events, total, on = "Group", how = "inner")
        result[f"P({y.name}|Group)"] = result["Count"] / result["Sum"]
        result.drop(["Count", "Sum"], axis = 1, inplace=True)

        self.map = result
        
        for name in self.map:
            if name.endswith("|Group)"):
                self.p_name = name
            elif name != "Group":
                self.y_name = name        
        
        return self

    def predict(self, X: pd.Series, k: int = -1) -> List:
        
        def retrieve(x):
            new_x = sorted(x, key = lambda x: -x[1])
            return new_x[: k if k > 0 else len(x)]
        
        def unstack(X):
            new_X = pd.Series(X).explode().apply(pd.Series)
            new_X.columns = ["Group", "P(Group)"]
            new_X.reset_index(drop = False, inplace = True)
            return new_X
        
        merged = pd.merge(unstack(X), self.map, on = "Group", how = "inner")
        merged["Prob"] = merged["P(Group)"] * merged[self.p_name]
        
        result = merged.groupby(["index", "Topic"]).agg(Prob = ("Prob", "sum"))
        result.reset_index(drop = False, inplace = True)
        
        result["Pred"] = result[[self.y_name, "Prob"]].apply(lambda x: (x[0], x[1]), axis = 1)
        result = result.groupby("index").agg(**{self.y_name: ("Pred", retrieve)})
        
        return [x for x in result[self.y_name]]