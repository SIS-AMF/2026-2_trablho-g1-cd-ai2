# Com o CSV do Kaggle (https://www.kaggle.com/datasets/yasserh/titanic-dataset?resource=download), podemos aplicar feature engineering avançado com as colunas:
# Name → extrair o título social (Mr, Miss, Dr, etc.)
# Cabin → extrair o deck da cabine (A, B, C... ou U para desconhecido)
# Ticket → extrair prefixo do ticket (tipo PC, STON, etc.)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 1) Carregar o dataset do Kaggle
df = pd.read_csv("sample_data/titanic.csv")

# 2) Tratar dados faltantes
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 3) Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["IsChild"] = (df["Age"] < 12).astype(int)

# Título do nome
df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt','Col','Don', 'Dr', 
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Letra da cabine
df["CabinLetter"] = df["Cabin"].str[0].fillna("U")  # U = Unknown

# Prefixo do ticket
df["TicketPrefix"] = df["Ticket"].apply(lambda x: x.split()[0] if not x.isnumeric() else "NUM")

# Agrupamento por frequência de ticket
ticket_counts = df["Ticket"].value_counts()
df["TicketFreq"] = df["Ticket"].apply(lambda x: "Group" if ticket_counts[x] > 1 else "Solo")

# Binning (quantis) para Fare e Age
df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
df["AgeBin"] = pd.qcut(df["Age"], 4, labels=False)

# 4) One-hot encoding
df = pd.get_dummies(df, columns=[
    "Sex", "Embarked", "Title", "CabinLetter", "TicketPrefix", "TicketFreq", "FareBin", "AgeBin"
], drop_first=True)

# 5) Selecionar X e y
X = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]

# 6) Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=40
)

# 7) GridSearchCV com param_grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'random_state': [46,40,130,7777]
}

xgb = XGBClassifier(eval_metric="logloss", random_state=40)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="accuracy",
    n_jobs=-1,
    cv=cv,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Melhores parâmetros:", grid_search.best_params_)

# 8) Modelo final
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Acurácia com XGBoost:", accuracy_score(y_test, y_pred))

# 9) Visualizar importância das variáveis
importances = best_model.feature_importances_
features = np.array(X.columns)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Importância das variáveis (XGBoost)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()

# 10) Ensemble: Voting Classifier
ensemble = VotingClassifier(
    estimators=[
        ("xgb", best_model),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=40)),
        ("lr", LogisticRegression(max_iter=1000))
    ],
    voting="soft"
)

ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
print("Acurácia com Ensemble (Voting):", accuracy_score(y_test, y_pred_ens))


# O VotingClassifier combina 3 modelos com lógicas bem diferentes:
# XGBoost Boosting (aditivo)  Corrige erros residuais em árvores fracas
# RandomForest    Bagging (paralelo)  Reduz variância e overfitting
# LogisticRegression  Linear  Excelente em padrões lineares ou quase lineares
# Quando combinamos esses modelos, eles se corrigem mutuamente, especialmente com voting='soft', que usa a média das probabilidades.
# O ensemble aproveita o melhor de cada modelo e entrega um modelo mais generalizável.