from sklearn.pipeline import Pipeline
from topic_classification import LDAClassificationModel, LDATopicModel, LDATransformer

X = [
    """Mathematical finance, also known as quantitative finance and financial mathematics, 
    is a field of applied mathematics, concerned with mathematical modeling of financial markets.""",
    """chemistry, the science that deals with the properties, composition, and structure of 
    substances (defined as elements and compounds), the transformations they undergo, and 
    the energy that is released or absorbed during these processes. Every substance, whether 
    naturally occurring or artificially produced, consists of one or more of the hundred-odd 
    species of atoms that have been identified as elements.""",
    """Mathematics (from Ancient Greek μάθημα (máthēma) 'knowledge, study, learning') is an 
    area of knowledge, which includes the study of such topics as numbers (arithmetic and 
    number theory),[1] formulas and related structures (algebra),[2] shapes and spaces in 
    which they are contained (geometry),[1] and quantities and their changes (calculus and 
    analysis).[3][4][5] There is no general consensus about its exact scope or 
    epistemological status.""",
    """Organic chemistry is the study of the structure, properties, composition, reactions, 
    and preparation of carbon-containing compounds. Most organic compounds contain carbon 
    and hydrogen, but they may also include any number of other elements (e.g., nitrogen, 
    oxygen, halogens, phosphorus, silicon, sulfur).""",
]

y = ["matematica", "quimica", "matematica", "quimica"]


def yielder(X):
    for x in X:
        yield x


transformer = LDATransformer(no_below=1, lang="en")
transformer = transformer.fit(yielder(X))
X_t = transformer.transform(yielder(X))

topic_model = LDATopicModel(num_topics=2, random_state=0)
X_topic = topic_model.fit_transform(X_t)

clf_model = LDAClassificationModel()
clf_model.fit(X_topic, y)

preds = clf_model.predict(X_topic)

print(preds)

clf = Pipeline(
    [("transformer", transformer), ("LDA", topic_model), ("classifier", clf_model)]
)

new_text = """Algebra is the study of mathematical symbols and the rules for 
manipulating these symbols in formulas; it is a unifying thread of almost 
all of mathematics."""

new_pred = clf.predict(yielder([new_text]))

print(new_pred)
