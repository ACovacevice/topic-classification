from topic_classification import (
    DictTransformer,
    DocTransformer,
    LDAClassifier,
    LDATransformer,
)

X = [
    """
    Mathematical finance, also known as quantitative finance and financial mathematics, 
    is a field of applied mathematics, concerned with mathematical modeling of financial markets. 
    In general, there exist two separate branches of finance that require advanced quantitative techniques: 
    derivatives pricing on the one hand, and risk and portfolio management on the other.
    """,
    """
    Mathematics is an area of knowledge that includes the topics of numbers, formulas and related structures, 
    shapes and the spaces in which they are contained, and quantities and their changes. 
    These topics are represented in modern mathematics with the major subdisciplines of number 
    theory, algebra, geometry, and analysis, respectively.
    """,
    """
    Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the 
    mathematical study of continuous change, in the same way that geometry is the study of shape, 
    and algebra is the study of generalizations of arithmetic operations.
    """,
    """
    Chemistry is the scientific study of the properties and behavior of matter. 
    It is a natural science that covers the elements that make up matter to the compounds made of atoms, 
    molecules and ions: their composition, structure, properties, behavior and the changes they undergo 
    during a reaction with other substances.
    """,
    """
    Organic chemistry is a subdiscipline within chemistry involving the scientific study of the structure,
    properties, and reactions of organic compounds and organic materials, i.e., matter in its various forms 
    that contain carbon atoms. Study of structure determines their structural formula.
    """,
]

y = ["Mathematics", "Mathematics", "Mathematics", "Chemistry", "Chemistry"]

model = LDAClassifier(
    DocTransformer(),
    DictTransformer(no_below=1, no_above=1),
    LDATransformer(num_topics=4, random_state=0),
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
Inorganic chemistry deals with synthesis and behavior of inorganic and organometallic compounds. 
This field covers chemical compounds that are not carbon-based, which are the subjects of organic chemistry. 
The distinction between the two disciplines is far from absolute, as there is much overlap in the subdiscipline 
of organometallic chemistry."""

new_preds = model.predict([new_text], k=1)
new_topics = model.predict_topic([new_text], k=3)

print()
print("New input:\n   ", new_text)
print()
print("Prediction:\n   ", new_preds[0][0])
print()
print("Latent topics, in order of relevance:\n   ", new_topics)
