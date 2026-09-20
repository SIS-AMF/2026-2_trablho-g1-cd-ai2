import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1) Carregar e selecionar
titanic = sns.load_dataset("titanic")
df = titanic[["survived","pclass","sex","age","sibsp","parch","fare","embarked"]].copy()

# 2) Engenharia de atributos (row-wise → não aprende nada do conjunto, então não vaza)
df["familySize"]    = df["sibsp"] + df["parch"] + 1
df["farePerPerson"] = df["fare"] / df["familySize"]
df["isAlone"]       = (df["familySize"] == 1).astype(int)
df["isChild"]       = (df["age"] < 12).astype(int)

X = df.drop(columns=["survived"])
y = df["survived"]

# 3) Split 80/20 (holdout final intocado até o fim)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 4) Pré-processamento DENTRO do pipeline (sem vazamento)
num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

numeric = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ]
)

# 5) Modelo
gb = GradientBoostingClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=42
)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", gb)])

# 6) Cross-validation apenas no CONJUNTO DE TREINO (estimativa honesta)
cv_scores = cross_validate(
    pipe, X_train, y_train, cv=5,
    scoring=["accuracy","precision","recall","f1"],
    return_train_score=False
)

print("[CV no treino 80%]")
print("Accuracy:",  cv_scores["test_accuracy"].mean())
print("Precision:", cv_scores["test_precision"].mean())
print("Recall:",    cv_scores["test_recall"].mean())
print("F1:",        cv_scores["test_f1"].mean())

# 7) Treino final no treino inteiro (80%) e avaliação no TESTE (20%)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("\n[Desempenho no teste 20%]")
print("Accuracy:",  accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:",    recall_score(y_test, y_pred))
print("F1:",        f1_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))
