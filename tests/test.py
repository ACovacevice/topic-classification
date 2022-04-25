from sklearn.pipeline import Pipeline
from topic_classification import LDAClassifier, LDATransformer

X = [
    """A matemática financeira utiliza uma série de conceitos matemáticos 
    aplicados à análise de dados financeiros em geral. Os problemas clássicos 
    de matemática financeira são ligados a questão do valor do dinheiro no tempo 
    e como isso é aplicado a empréstimos, investimentos e avaliação financeira de projetos.""",
    """Química é a ciência que estuda a composição, estrutura, propriedades da matéria, as 
    mudanças sofridas por ela durante as reações químicas e a sua relação com a energia.""",
    """A matemática (dos termos gregos μάθημα, transliterado máthēma, 'ciência', conhecimento' 
    ou 'aprendizagem';[1] e μαθηματικός, transliterado mathēmatikós, 
    'inclinado a aprender') é a ciência do raciocínio lógico e abstrato, que estuda 
    quantidades (teoria dos números), espaço e medidas (geometria), estruturas, 
    variações[2] e estatística.[3][4][5] Não há, porém, uma definição consensual por parte 
    da comunidade científica.[6][7] O trabalho matemático consiste em procurar e relacionar 
    padrões,[8][9] de modo a formular conjecturas[10] cuja veracidade ou falsidade é 
    provada por meio de deduções rigorosas a partir de axiomas e definições. 
    A matemática desenvolveu-se principalmente na Mesopotâmia, no Egito, na Grécia, na 
    Índia e no Oriente Médio. A partir da Renascença, o desenvolvimento da matemática 
    intensificou-se na Europa, quando novas descobertas científicas levaram a um crescimento 
    acelerado que dura até os dias de hoje.""",
    """A química orgânica é a parte do campo do conhecimento que estuda todos os compostos 
    que têm em sua base a estrutura de átomos de carbono e outros elementos presentes em 
    organismos vivos, tanto do reino animal quanto do reino vegetal, como hidrogênio, 
    oxigênio, nitrogênio, entre outros.""",
]

y = ["matematica", "quimica", "matematica", "quimica"]

transformer = LDATransformer(no_below=1, lang="pt")
X_t = transformer.fit_transform(X)

clf_model = LDAClassifier(num_topics=3, random_state=0)
clf_model.fit(X_t, y)

preds = clf_model.predict(X_t)

print(preds)

preds = clf_model.predict(X_t, k=1)

print(preds)
print("Weighted F1:", clf_model.score(X_t, y))

clf = Pipeline([("transformer", transformer), ("classifier", clf_model)])

new_text = """Álgebra é o ramo da matemática que estuda a manipulação formal de equações,
operações matemáticas, polinômios e estruturas algébricas. A álgebra é um dos principais
ramos da matemática pura, juntamente com a geometria, topologia, análise, e teoria dos
números."""

new_pred = clf.predict([new_text])

print(new_pred)
