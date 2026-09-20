import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Carregar dataset Titanic
titanic = sns.load_dataset("titanic")

# Selecionar variáveis úteis
df = titanic[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()

# Tratar dados faltantes (cuidado! vazamento de dados por usar a média do dataset inteiro!)
df["age"] = df["age"].fillna(df["age"].median()) 
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0]) 

# ===== Novas features =====
df["familySize"] = df["sibsp"] + df["parch"] + 1
df["farePerPerson"] = df["fare"] / df["familySize"]
df["isAlone"] = (df["familySize"] == 1).astype(int)
df["isChild"] = (df["age"] < 12).astype(int)

# Transformar variáveis categóricas em numéricas (one-hot)
df = pd.get_dummies(df, drop_first=True)

# 2) Correlação das variáveis com "survived"
corr = df.corr()["survived"].sort_values(ascending=False)
print("Correlação com sobrevivência:")
print(corr)

# Visualização
plt.figure(figsize=(6,4))
corr.drop("survived").plot(kind="bar")
plt.title("Correlação das variáveis com sobrevivência")
plt.show()

# 3) Separar treino e teste
X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# from sklearn.impute import SimpleImputer
# imputer_age = SimpleImputer(strategy="median")
# imputer_embarked = SimpleImputer(strategy="most_frequent")

# fit -> treino
# transform -> teste
# X_train["age"] = imputer_age.fit_transform(X_train[["age"]])
# X_test["age"] = imputer_age.transform(X_test[["age"]])

# X_train["embarked"] = imputer_embarked.fit_transform(X_train[["embarked"]])
# X_test["embarked"] = imputer_embarked.transform(X_test[["embarked"]])

# agora sim feature engineering sem vazamento, com variáveis que dependem de age e embarked, por exemplo:
# df["isChild"] = (df["age"] < 12).astype(int)
# ...

# 4) Modelo RandomForest
model = RandomForestClassifier(
	n_estimators=300, # mais árvores → mais robusto
	n_jobs=-1,
	# class_weight="balanced",
	random_state=42,
	# ... teste diferentes hiperparâmetros do Random Forest!	
)
# teste o GradientBoostingClassifier também!

model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

# Importância das variáveis
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# Visualizar
plt.figure(figsize=(8,4))
importances.plot(kind="bar")
plt.title("Importância das variáveis (RandomForest)")
plt.show()


# === Relatório completo ===
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# === Matriz de confusão ===
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Morreu", "Sobreviveu"], yticklabels=["Morreu", "Sobreviveu"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()
