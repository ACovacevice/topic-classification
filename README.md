# LDA for Document Classification

This package consists of a textual document classification model that is based on the <a href="https://radimrehurek.com/gensim/models/ldamulticore.html">Latent Dirichlet Allocation (LDA)</a> algorithm for topic modeling.
It is a prototype that is memory-intensive, so it is not recommended for huge datasets.

## Methodology

The LDA algorithm allows for the estimation of <b>latent groups</b> based on the co-occurrence of tokens (words or n-grams) found in textual data. For example, by inputting the two following documents

```python
corpus = [
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
```

the algorithm would be able to assert the co-occurrence of the words

- `Mathematics`, `numbers`, `formulas`, `algebra`, and `geometry` (Group 1);
- `Biology`, `life`, `natural`, `organisms`, and `cells` (Group 2);

providing two distinct <b>latent groups</b> as the result -- each being described by their words (so it might not be so "latent" after all).

It consists of an unsupervised learning process. However, we can make a supervised classification model by including labeled data in the training process.

Let's say the labels were

```python
label = ["Mathematics", "Biology"]
```

The probability of a new document,

```python
new_doc = """
Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the mathematical
study of continuous change, in the same way that geometry is the study of shape, and algebra is the study
of generalizations of arithmetic operations.
"""
```

being classified as either `Mathematics` or `Biology` is given by Bayes' Theorem:

- <i>P(Mathematics) = P(Mathematics|Group 1) _ P(Group 1) + P(Mathematics|Group 2) _ P(Group 2)</i>

- <i>P(Biology) = P(Biology|Group 1) _ P(Group 1) + P(Biology|Group 2) _ P(Group 2)</i>

Each probability P(<i>Label</i>|<i>Group</i>) is computed and stored as model parameters during the training step.

Predictions tend to be more precise the richer the vocabulary is. However, one must optimize for the number of latent groups to obtain the best model.

## Usage

Please, refer to our <a href="https://github.com/ACovacevice/topic-classification/tree/main/examples">examples</a>.

Be sure to tune the parameters of `DictTransformer` to adjust the size of the vocabulary you intend on using.

You can visualize the latent groups created in a nice interface by using <a href="https://pypi.org/project/pyLDAvis/">pyLDAvis</a>.
