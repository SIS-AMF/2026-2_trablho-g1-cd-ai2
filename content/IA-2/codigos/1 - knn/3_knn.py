# pip install scikit-learn matplotlib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 1) Dados: cada imagem 8x8 -> vetor de 64 features
X, y = load_digits(return_X_y=True)

# 2) Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 3) Pipeline: normalização + KNN (k=3, distância euclidiana)
model = make_pipeline(
    StandardScaler(),                  # KNN é sensível à escala
    KNeighborsClassifier(n_neighbors=3, metric="euclidean")
)

# 4) Treino e avaliação
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, digits=3))

# 5) Matriz de confusão
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("KNN (k=3) - Digits")
plt.show()

# 6) Mostrar alguns exemplos previstos
idx = np.random.RandomState(7).choice(len(X_test), size=6, replace=False)
plt.figure(figsize=(8,3))
for i, j in enumerate(idx, 1):
    plt.subplot(1, 6, i)
    plt.imshow(X_test[j].reshape(8, 8), cmap="gray")
    plt.axis("off")
    plt.title(f"pred: {y_pred[j]}\ntrue: {y_test[j]}")
plt.suptitle("Exemplos de predições (KNN)")
plt.tight_layout()
plt.show()
