import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1) Ler CSV exportado do Google Forms
# =========================
df = pd.read_csv("respostas.csv")

print("Colunas encontradas:")
print(df.columns.tolist())

# =========================
# 2) Renomear colunas
# Ajuste os nomes àqueles que vierem do seu Google Forms
# =========================
df = df.rename(columns={
    "Nome": "nome",
    "Em decisões importantes, você tende a:": "decisao_analise",
    "Em um desafio, você prefere:": "risco",
    "Quando algo dá errado, você tende a:": "persistencia",
    "Para resolver um problema, você prefere:": "colaboracao",
    "Em atividades e projetos, você se identifica mais com:": "criatividade",
    "Seu estilo costuma ser mais:": "planejamento",
    "Em situações novas ou incertas, você se sente:": "incerteza"
})

# =========================
# 3) Selecionar variáveis para clustering
# =========================
features = [
    "decisao_analise",
    "risco",
    "persistencia",
    "colaboracao",
    "criatividade",
    "planejamento",
    "incerteza"
]

X = df[features].copy()

# Garantir que tudo é numérico
for col in features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remover linhas com valores faltantes
dados_validos = X.notna().all(axis=1)
df = df[dados_validos].copy()
X = X[dados_validos].copy()

print("\nNúmero de respostas válidas:", len(df))

# =========================
# 4) Padronizar dados
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5) Rodar K-means
# =========================
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

# =========================
# 6) Reduzir para 2 dimensões para visualização
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

# =========================
# 7) Plotar clusters
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(df["pca1"], df["pca2"], c=df["cluster"])

for _, row in df.iterrows():
    plt.text(row["pca1"] + 0.03, row["pca2"] + 0.03, str(row["nome"]), fontsize=8)

plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Clusters da turma")
plt.grid(True)
plt.show()

# =========================
# 8) Mostrar participantes por cluster
# =========================
print("\nParticipantes e clusters:")
print(df[["nome", "cluster"]].sort_values("cluster"))


# dado cluster, pegar um membro aleatório e pedir qual perfil mais se encaixa:
# estrategistas
# experimentadores
# criativos colaborativos
# O K-means é uma técnica de clustering não supervisionada, o que significa que ele agrupa os dados com base em suas características, mas não fornece rótulos ou interpretações automáticas para esses grupos. A interpretação dos clusters é um processo subjetivo e requer uma análise cuidadosa das características médias de cada cluster, bem como do contexto dos dados.

# =========================
# 9) Média das características por cluster
# =========================
print("\nPerfil médio de cada cluster:")
print(df.groupby("cluster")[features].mean().round(2))
print(pca.components_) # como cada variável original contribui para cada eixo do PCA




import matplotlib.pyplot as plt

dados_plot = df.set_index("nome")[features]

plt.figure(figsize=(10, 8))

plt.imshow(dados_plot, aspect="auto")

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(dados_plot.index)),
    dados_plot.index
)

plt.colorbar(label="Valor")

plt.title("Perfil multidimensional dos alunos")
plt.tight_layout()

plt.show()