import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ---------- Criando um dataset simples ----------
# Exemplo: Classificar se uma pessoa joga videogame (sim/não)
# baseado em idade e tempo livre por semana

data = {
    "idade": [15, 25, 32, 40, 50, 65, 12, 22, 28, 35],
    "horas_livres": [30, 10, 5, 8, 2, 1, 25, 15, 7, 4],
    "joga": ["sim", "sim", "não", "não", "não", "não", "sim", "sim", "não", "não"]
}
#data = {
#    "idade": [15, 25, 32, 40, 50, 65, 12, 22, 28, 35, 45, 18],
#    "horas_livres": [30, 10, 5, 8, 2, 1, 25, 15, 7, 4, 20, 3],
#    "joga": ["sim", "sim", "não", "não", "não", "não", "sim", "sim", "não", "não", "sim", "não"]
#}

df = pd.DataFrame(data)

X = df[["idade", "horas_livres"]]   # features
y = df["joga"]                      # target

# ---------- Treinando a árvore ----------
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X, y)

# ---------- Fazendo uma predição ----------
# Exemplo: pessoa de 20 anos com 12 horas livres
exemplo = [[20, 12]]
pred = clf.predict(exemplo)
print(f"Predição para idade=20, horas_livres=12 → {pred[0]}")

# ---------- Visualizando a árvore ----------
plt.figure(figsize=(8,6))
plot_tree(clf, feature_names=["idade", "horas_livres"], class_names=["não", "sim"], filled=True)
plt.show()
