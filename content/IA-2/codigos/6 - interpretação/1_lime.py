import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# 1) Dataset Titanic
titanic = sns.load_dataset("titanic")

# Seleção de variáveis
df = titanic[["survived", "pclass", "sex", "age", "fare", "embarked"]].dropna()

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Treinar modelo
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# 3) Escolher instância para explicar
i0 = 1
x0 = X_test.iloc[[i0]]

print("Previsão do modelo:", rf.predict(x0)[0],
      "| Probabilidade de sobreviver:", rf.predict_proba(x0)[0,1])

# 4) Criar perturbações ao redor do passageiro
n_samples = 1500
noise = np.random.normal(0, X_train.std(axis=0)*0.5, size=(n_samples, X_train.shape[1]))
Z = x0.values + noise
Z = np.clip(Z, X_train.min().values, X_train.max().values)

# 5) Obter previsões do modelo
Z_proba = rf.predict_proba(Z)[:,1]

# 6) Pesos pela proximidade
dists = cosine_distances(Z, x0)
kernel_width = np.sqrt(X_train.shape[1])
weights = np.exp(-(dists**2)/(kernel_width**2))

# 7) Ajustar modelo linear local
scaler = StandardScaler().fit(Z)
Z_std = scaler.transform(Z)

ridge = Ridge(alpha=1.0)
ridge.fit(Z_std, Z_proba, sample_weight=weights.ravel())

# Importâncias locais
coefs = ridge.coef_
importance = pd.Series(coefs, index=X_train.columns).sort_values(key=lambda s: abs(s), ascending=False)

print("\nImportância local (LIME-style):")
print(importance.head())

# 8) Gráfico
importance.head(6).plot(kind="barh", figsize=(6,4))
plt.title("Explicação local (LIME-style) para o passageiro")
plt.show()

# ============================================================
# 9. LIME utilizando biblioteca
# ============================================================

from lime.lime_tabular import LimeTabularExplainer

# Nomes das variáveis utilizadas pelo modelo
features = X_train.columns.tolist()

# Criar o explicador LIME
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=features,
    class_names=["Não sobreviveu", "Sobreviveu"],
    mode="classification",
    random_state=42
)

# Utilizar o mesmo passageiro da explicação manual
i = i0

passageiro = X_test.iloc[i]

print("\nPassageiro escolhido:")
print(passageiro)

print("\nClasse real:")
print(
    "Sobreviveu"
    if y_test.iloc[i] == 1
    else "Não sobreviveu"
)

print("\nPredição do modelo:")

pred = rf.predict(
    passageiro.to_frame().T
)[0]

print(
    "Sobreviveu"
    if pred == 1
    else "Não sobreviveu"
)

print("\nProbabilidades:")

probas = rf.predict_proba(
    passageiro.to_frame().T
)[0]

print("Não sobreviveu:", probas[0])
print("Sobreviveu:", probas[1])


# Explicação LIME
explicacao_lime = lime_explainer.explain_instance(
    data_row=passageiro.values,
    predict_fn=rf.predict_proba,
    num_features=len(features)
)

print("\nExplicação LIME:")

for regra, peso in explicacao_lime.as_list():
    print(regra, "->", peso)


# Gráfico
explicacao_lime.as_pyplot_figure()

plt.title("LIME — explicação da previsão")

plt.tight_layout()
plt.show()