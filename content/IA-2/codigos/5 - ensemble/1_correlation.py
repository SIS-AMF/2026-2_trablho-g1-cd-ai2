import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 1) Carregar dataset Titanic
titanic = sns.load_dataset("titanic")

# Selecionar variáveis úteis
df = titanic[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()

# Tratar dados faltantes
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