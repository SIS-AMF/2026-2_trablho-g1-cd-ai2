import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# =========================
# 1) Obter dataset
# =========================
df = pd.read_csv("rh_turnover_200.csv")
print("Primeiras linhas:")
print(df.head())

print("\nValores faltantes:")
print(df.isna().sum())

# =========================
# 2) Preparar os dados + feature engineering
# =========================
mapa_salario = {"baixo": 0, "medio": 1, "alto": 2}
df["salario"] = df["salario"].map(mapa_salario)
# label_encoder = LabelEncoder()
# X["salario"] = label_encoder.fit_transform(X["salario"])

# funcionário com sobrecarga
df["sobrecarga"] = (df["horas_mes"] > 210).astype(int)

# funcionário sem promoção
df["sem_promocao"] = (df["promocoes"] == 0).astype(int)

print("\nBase após feature engineering:")
print(df.head())

X = df[["tempo_empresa", "satisfacao", "horas_mes", "sobrecarga", "sem_promocao", "promocoes", "salario"]] # df.drop(columns=["saiu"])
y = df["saiu"]

# =========================
# 3) Divisão treino-teste
# =========================
# stratify=y é um parâmetro do train_test_split que faz a divisão preservando a proporção das classes da variável alvo (para train e test).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4) Tratamento de faltantes + normalização
# =========================
# imputação SEM vazamento (apenas no treino)
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# print("\nExiste NaN no treino?", pd.isna(X_train).any())
# print("Existe NaN no teste?", pd.isna(X_test).any())

# normalização
# nota: decision tree não precisa de scaler; não dependem de distância, como o KNN; não dependem de margem geométrica, como o SVM; só ficam testando cortes do tipo feature <= valor.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 5) Treinar o modelo de ML
# =========================
model = KNeighborsClassifier(n_neighbors=3)
# model = DecisionTreeClassifier(max_depth=3, random_state=42)
# model = SVC(kernel="linear", probability=True, random_state=42)

model.fit(X_train, y_train)

# Desempenho no treinamento
y_pred_train = model.predict(X_train)
print("\n=== Avaliação TRAIN ===")
print("Acurácia:", accuracy_score(y_train, y_pred_train))

print("\nMatriz de confusão:")
cm = confusion_matrix(y_train, y_pred_train)
print(confusion_matrix(y_train, y_pred_train))

# =========================
# 6) Avaliar o modelo
# =========================
y_pred = model.predict(X_test)

print("\n=== Avaliação TEST ===")
print("Acurácia:", accuracy_score(y_test, y_pred))

print("\nMatriz de confusão:")
cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Permaneceu", "Saiu"]
)
disp.plot()
plt.title("Matriz de Confusão")
plt.show()

print("\nRelatório:")
print(classification_report(y_test, y_pred))


# =========================
# Visualizar árvore (quando decision tree)
# =========================
# from sklearn.tree import plot_tree
# plt.figure(figsize=(18, 10))
# plot_tree(
#     model,
#     feature_names=X.columns,
#     class_names=["Permaneceu", "Saiu"],
#     filled=True,
#     rounded=True,
#     fontsize=10
# )
# plt.title("Árvore de Decisão")
# plt.show()

# =========================
# 8) Salvar o modelo treinado
# =========================
joblib.dump(model, "trained_model.joblib")

# ...

# =========================
# 9) Carregar o modelo treinado (pode ser feito em outro script)
# =========================
model = joblib.load("trained_model.joblib")

# =========================
# 10) Inferir no modelo
# (prever os 3 colaboradores)
# =========================
novos_colaboradores = pd.DataFrame([
    {
        "nome": "Ana",
        "tempo_empresa": 1,
        "satisfacao": 0.28,
        "horas_mes": 235,
        "promocoes": 0,
        "salario": "baixo"
    },
    {
        "nome": "Carlos",
        "tempo_empresa": 6,
        "satisfacao": 0.89,
        "horas_mes": 176,
        "promocoes": 2,
        "salario": "alto"
    },
    {
        "nome": "João",
        "tempo_empresa": 5,
        "satisfacao": 0.53,
        "horas_mes": 215,
        "promocoes": 1,
        "salario": "alto"
    }
])

novos_colaboradores["sobrecarga"] = (
    novos_colaboradores["horas_mes"] > 210
).astype(int)

novos_colaboradores["sem_promocao"] = (
    novos_colaboradores["promocoes"] == 0
).astype(int)

# aplicar mesmo encoder
novos_colaboradores["salario"] = novos_colaboradores["salario"].map(mapa_salario)

# novos_colaboradores = imputer.transform(novos_colaboradores) # se tivesse que inputar valores faltantes
novos_scaled = scaler.transform(novos_colaboradores.drop(columns=["nome"]))

# predição
predicoes = model.predict(novos_scaled)
probabilidades = model.predict_proba(novos_scaled)[:, 1]

# resultado final
resultado = novos_colaboradores.copy()
resultado["classe_prevista"] = predicoes
resultado["prob_saida"] = probabilidades

resultado["classe_prevista"] = resultado["classe_prevista"].map({
    0: "Permaneceu",
    1: "Saiu"
})

print(resultado)