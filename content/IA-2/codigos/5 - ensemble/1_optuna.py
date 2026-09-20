import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1) Carregar dados do Kaggle
df = pd.read_csv("sample_data/titanic.csv")

# 2) Pré-processamento básico
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Engenharia de atributos
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["IsChild"] = (df["Age"] < 12).astype(int)
df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt','Col','Don', 'Dr',
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
df["CabinLetter"] = df["Cabin"].str[0].fillna("U")
df["TicketPrefix"] = df["Ticket"].apply(lambda x: x.split()[0] if not x.isnumeric() else "NUM")
ticket_counts = df["Ticket"].value_counts()
df["TicketFreq"] = df["Ticket"].apply(lambda x: "Group" if ticket_counts[x] > 1 else "Solo")
df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
df["AgeBin"] = pd.qcut(df["Age"], 4, labels=False)

# One-hot encoding
df = pd.get_dummies(df, columns=[
    "Sex", "Embarked", "Title", "CabinLetter", "TicketPrefix", "TicketFreq", "FareBin", "AgeBin"
], drop_first=True)

# Separar X e y
X = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]

# Dividir treino/teste fixo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Função objetivo do Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    return score

# 4) Rodar Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# 5) Melhor modelo encontrado
print("Melhores parâmetros encontrados:", study.best_params)

# 6) Treinar modelo final
best_model = XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)

# 7) Avaliação final
y_pred = best_model.predict(X_test)
print("Acurácia com XGBoost otimizado (Optuna):", accuracy_score(y_test, y_pred))
