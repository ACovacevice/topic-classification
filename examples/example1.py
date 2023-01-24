from topic_classification import (
    DictTransformer,
    DocTransformer,
    LDAClassifier,
    LDATransformer,
)

X = [
    """
    Mathematics is an area of knowledge that includes the topics of numbers, formulas and related structures,
    shapes and the spaces in which they are contained, and quantities and their changes. These topics are
    represented in modern mathematics with the major subdisciplines of number theory, algebra, geometry,
    and analysis, respectively.
    """,
    """
    Biology is the scientific study of life. It is a natural science with a broad scope but has several
    unifying themes that tie it together as a single, coherent field. For instance, all organisms are made
    up of cells that process hereditary information encoded in genes, which can be transmitted to future
    generations.
    """
]

y = ["Mathematics", "Biology"]

model = LDAClassifier(
    DocTransformer(),
    DictTransformer(no_below=1, no_above=1),
    LDATransformer(num_topics=2, random_state=0),
)

model = model.fit(X, y)
preds = model.predict(X, k=1)
score = model.score(X, y)

print()
print("Predictions:")
print("    y_true{0:>9}y_pred".format(""))
for y_true, y_pred in zip(y, preds):
    print("    {0:>11}{1:>15}".format(y_true, y_pred[0]))

print()
print(f"Train set weighted F1-Score: {score}")

new_text = """
Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the mathematical study of continuous change, in the same way that geometry is the study of shape, and algebra is the study of generalizations of arithmetic operations."""

new_preds = model.predict([new_text], k=1)
new_topics = model.predict_topic([new_text], k=3)

print()
print("New input:\n   ", new_text)
print()
print("Prediction:\n   ", new_preds[0][0])
print()
print("Latent topics, in order of relevance:\n   ", new_topics)