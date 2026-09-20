import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Dataset
comidas = [
    ("Hambúrguer", 500, 3, "Lanche"),
    ("Pizza",      700, 4, "Lanche"),
    ("Coxinha",    300, 3, "Lanche"),
    ("Pastel",     450, 4, "Lanche"),

    ("Feijoada",   900, 5, "Prato pesado"),
    ("Lasanha",    800, 3, "Prato pesado"),
    ("Churrasco",  850, 4, "Prato pesado"),
    ("Strogonoff", 750, 2, "Prato pesado"),

    ("Sorvete",    250, 1, "Sobremesa"),
    ("Bolo",       400, 2, "Sobremesa"),
    ("Pudim",      350, 1, "Sobremesa"),
    ("Mousse",     300, 1, "Sobremesa"),
]

# Features (calorias, tempero) e classes
X = np.array([[c[1], c[2]] for c in comidas], dtype=float)
y = np.array([c[3] for c in comidas])

# Modelo SVM RBF com normalização
clf = make_pipeline(
    StandardScaler(), 
    # SVC(kernel="linear", C=1.0, gamma="scale", random_state=42)
    # SVC(kernel="poly", C=1.0, gamma="scale", random_state=42)
    SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    # SVC(kernel="rbf", C=100.0, gamma=1.0, random_state=42)
)
clf.fit(X, y)  

# Previsões nos próprios dados (só para ilustrar)
pred = clf.predict(X)
for nome, real, p in zip([c[0] for c in comidas], y, pred):
    print(f"{nome:12s}  real={real:12s}  pred={p:12s}")

# Predição exemplo
novo = np.array([[620, 3]])
pred_novo = clf.predict(novo)[0]
print("Novo prato (620 kcal, tempero 3) →", pred_novo)

######## Plot
# Encoder para mapear rótulos -> inteiros (para o gráfico)
le = LabelEncoder()
y_int = le.fit_transform(y)   # ex.: {'Lanche':0, 'Prato pesado':1, 'Sobremesa':2}

# Grade para fronteira
x_min, x_max = X[:,0].min() - 50, X[:,0].max() + 50
y_min, y_max = X[:,1].min() - 1,  X[:,1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

# Predição no grid e transformação para inteiros
Z_labels = clf.predict(np.c_[xx.ravel(), yy.ravel()])     # strings
Z = le.transform(Z_labels).reshape(xx.shape)              # inteiros (OK para contourf)

# Plot das regiões + pontos
n_cls = len(le.classes_)
custom_colors = ["#377eb8", "#e41a1c", "#4daf4a"]  # blue, red, green
cmap = mcolors.ListedColormap(custom_colors[:n_cls])
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_cls+0.5, 1), ncolors=n_cls)

plt.figure(figsize=(7,6))

# regiões (usa Z com ints)
plt.contourf(xx, yy, Z, alpha=0.25, cmap=cmap, norm=norm)
# pontos rotulados com o MESMO cmap/norm
plt.scatter(X[:,0], X[:,1], c=y_int, s=80, edgecolor='k', alpha=0.9, cmap=cmap, norm=norm)

# nomes nos pontos
nomes = [c[0] for c in comidas]
for nome, (cal, temp) in zip(nomes, X):
    # Deslocamento visual para o rótulo não ficar em cima do ponto:
    plt.annotate(nome, (cal, temp), textcoords="offset points", xytext=(5,5), ha="left", fontsize=8)

plt.scatter(novo[0,0], novo[0,1], s=160, marker="X", edgecolor="k", linewidths=1.5)  # ponto destacado
plt.annotate(f"Novo: {pred_novo}", (novo[0,0], novo[0,1]), textcoords="offset points", xytext=(8,8), ha="left", fontsize=9)

plt.xlabel("Calorias (kcal)")
plt.ylabel("Nível de tempero (0–5)")
plt.title("SVM RBF – Fronteira de decisão (comidas)")
# legenda manual com o mapeamento do encoder
handles = [plt.Line2D([0],[0], marker='o', linestyle='', label=cls) for cls in le.classes_]
plt.legend(handles=handles, title="Classe", loc="best")
plt.tight_layout()

# --- Legenda com as MESMAS cores ---
handles = []
for i, cls in enumerate(le.classes_):
    h = plt.Line2D([0],[0], marker='o', linestyle='',
                   markerfacecolor=cmap(i), markeredgecolor='k', label=cls)
    handles.append(h)
plt.legend(handles=handles, title="Classe", loc="best")

plt.tight_layout()
plt.show()

##### Show misclassified items
errs = [(n, r, p) for n, r, p in zip([c[0] for c in comidas], y, pred) if r != p]
if errs:
    print("\nMisclassified:")
    for n, r, p in errs:
        print(f"  {n}: real={r} pred={p}")
else:
    print("\nAll training points classified correctly.")