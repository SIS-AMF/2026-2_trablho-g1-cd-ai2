# Simple, practical SVM demo with scikit-learn
# Dataset: two interleaving half-moons (not linearly separable)
# We train two SVMs (linear vs RBF) and visualize decision boundaries + report accuracy.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1) Data
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2) Train two SVMs
svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# 3) Evaluate
y_pred_lin = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

acc_lin = accuracy_score(y_test, y_pred_lin)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"Acurácia (SVM Linear): {acc_lin:.3f}")
print(f"Acurácia (SVM RBF):    {acc_rbf:.3f}")

# 4) Helper to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    # grid
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # plot (1 chart per plot, no specific colors)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.scatter(X[:,0], X[:,1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

# 5) Visualize
plot_decision_boundary(svm_linear, X_test, y_test, "SVM Linear – fronteira de decisão (teste)")
plot_decision_boundary(svm_rbf,   X_test, y_test, "SVM RBF – fronteira de decisão (teste)")

# 6) Example prediction
example = np.array([[1.5, -0.4]])  # a single sample
pred_lin = svm_linear.predict(example)[0]
pred_rbf = svm_rbf.predict(example)[0]
print(f"Previsão do ponto {example.tolist()} -> Linear: {pred_lin}, RBF: {pred_rbf}")
