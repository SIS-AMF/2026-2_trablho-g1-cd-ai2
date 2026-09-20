import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt

# ---------- Dataset que exige ambas as features ----------
rows = [
    # jovem (<30): precisa >=10 h livres para "sim"
    (18,  5, "não"), (22,  8, "não"), (25, 12, "sim"), (19, 15, "sim"),
    (28,  9, "não"), (16, 11, "sim"), (24, 10, "sim"), (21, 14, "sim"),
    # adulto (>=30): precisa >=20 h livres para "sim"
    (30, 10, "não"), (35, 18, "não"), (40, 22, "sim"), (45, 25, "sim"),
    (33, 19, "não"), (50, 21, "sim"), (60, 17, "não"), (38, 24, "sim"),
    # casos de "fronteira" pra confundir se usar só uma feature
    (27, 20, "sim"),  # jovem com muitas horas -> sim
    (32, 12, "não"),  # adulto com poucas horas -> não
    (29,  9, "não"),  # jovem com poucas horas -> não
    (41, 20, "sim"),  # adulto com 20 horas -> sim
]

df = pd.DataFrame(rows, columns=["idade", "horas_livres", "joga"])
X = df[["idade", "horas_livres"]]
y = df["joga"]

# ---------- Árvore ----------
clf = DecisionTreeClassifier(
    criterion="entropy",    # ou "gini"
    max_depth=3,            # suficiente pra mostrar ambos os splits
    random_state=42
)
clf.fit(X, y)

# ---------- Importância das features ----------
print("Importâncias:", dict(zip(X.columns, clf.feature_importances_)))

# ---------- Árvore em texto (pra ver os splits usados) ----------
print(export_text(clf, feature_names=list(X.columns)))

# ---------- Predição de exemplo ----------
ex = [[26, 9], [26, 12], [42, 18], [42, 22]]  # 4 casos contrastantes
pred = clf.predict(ex)
for (i, h), p in zip(ex, pred):
    print(f"idade={i:2d}, horas_livres={h:2d} → {p}")


# ---------- Visualização ----------
plt.figure(figsize=(9,6))
plot_tree(clf, feature_names=["idade", "horas_livres"],
          class_names=["não", "sim"], filled=True)
plt.tight_layout()
plt.show()
