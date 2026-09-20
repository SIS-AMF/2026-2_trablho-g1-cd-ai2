import seaborn as sns
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
i0 = 10
x0 = X_test.iloc[[i0]]

print("Previsão do modelo:", rf.predict(x0)[0],
      "| Probabilidade de sobreviver:", rf.predict_proba(x0)[0,1])

# Converter para float64
X_train = X_train.astype("float64")
X_test = X_test.astype("float64")

# 5) SHAP
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# 6) Corrigir forma: usar apenas valores da classe 1
shap_vals_instance = shap_values[i0].values[:, 1]
feature_names = X_test.columns

# 7) Visualização estilo SHAP
shap_series = pd.Series(shap_vals_instance, index=feature_names)
shap_series = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)

print("\nImportância local (SHAP-style):")
print(shap_series.head())

shap_series.head(6).plot(kind="barh", figsize=(6, 4))
plt.title("Explicação local (SHAP-style) para o passageiro")
plt.xlabel("Valor SHAP")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
