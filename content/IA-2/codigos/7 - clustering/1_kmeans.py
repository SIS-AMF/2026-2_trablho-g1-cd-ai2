import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Create synthetic dataset
# np.random.seed(42)
clients = pd.DataFrame({
    'Name': ['Ana', 'Bruno', 'Carla', 'Diego', 'Eduarda', 'Fábio', 'Gabi', 'Henrique'],
    
    # Escala 1–5
    'Apimentado': [4, 2, 1, 5, 3, 2, 4, 3],
    'Leve': [3, 2, 5, 1, 3, 4, 2, 3],
    'Veggie': [2, 1, 5, 1, 2, 4, 3, 2],
    'Fitness': [3, 2, 5, 1, 3, 4, 3, 3],

    # Escalas diferentes
    'CaloriasMediaDia': [1800, 2500, 1600, 3200, 2000, 2700, 2200, 1900],
    # 'CaloriasMediaDia': np.concatenate([
    #    np.random.normal(1800, 100, n),  # Gera n valores, a partir de uma distribuição normal (gaussiana), com média 1800 e desvio padrão 100 
    #    np.random.normal(2800, 150, n),
    #    np.random.normal(1600, 100, n)
    # ]),
    'FrequenciaPedidoMes': [12, 15, 4, 10, 6, 5, 2, 11],

    # Variável booleana
    'IntoleranciaLactose': [0, 0, 1, 0, 1, 0, 1, 0]
    # 'IntoleranciaLactose': np.random.randint(0, 2, n) # n números inteiros aleatórios entre 0 (inclusive) e 2 (exclusivo)
})

# Step 2: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clients.drop("Name", axis=1))

# Step 3: Fit K-Means
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
clients['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Plot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clients['Cluster'], cmap="tab10")
for i, name in enumerate(clients['Name']):
    plt.text(X_pca[i,0]+0.05, X_pca[i,1]+0.05, name, fontsize=9)
plt.title("Customer Segmentation - Food Profile Clustering (K-Means)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

##########

new_client = pd.DataFrame({
    'Apimentado': [4],
    'Leve': [1],
    'Veggie': [3],
    'Fitness': [2],
    'CaloriasMediaDia': [2000],
    'FrequenciaPedidoMes': [8],
    'IntoleranciaLactose': [1]
})
new_scaled = scaler.transform(new_client)
predicted_cluster = kmeans.predict(new_scaled)[0]
cluster_names = {
    0: "C1",
    1: "C2",
    2: "C3",
    3: "C4",
}
print(f"🍽️ Este novo cliente pertence ao cluster {predicted_cluster}")

new_pca = pca.transform(new_scaled)
plt.figure(figsize=(9,7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clients['Cluster'], cmap='Set2', s=100, edgecolor='k')
for i, nome in enumerate(clients['Name']):
    cluster_label = cluster_names[clients.loc[i, 'Cluster']]
    plt.text(X_pca[i, 0]+0.05, X_pca[i, 1]+0.05, f"{nome}\n({cluster_label})", fontsize=8)
plt.scatter(new_pca[0, 0], new_pca[0, 1], c='black', s=120, marker='X', label='Novo Cliente')
plt.text(new_pca[0, 0]+0.05, new_pca[0, 1]+0.05, f"Novo\n({cluster_names[predicted_cluster]})", fontsize=9, color='black')
plt.title("Segmentação de Clientes - Perfil Alimentar")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def recomendar_menu(cluster_id):
    recomendacoes = {
        0: "🥗 Recomendado: Salada fresca com proteína leve, suco detox e sobremesa vegana.",
        1: "🌮 Recomendado: Tacos apimentados com opção sem lactose e bebida energética.",
        2: "🍱 Recomendado: Bento box com legumes cozidos, arroz integral e sobremesa leve.",
        3: "🍔 Recomendado: Hambúrguer artesanal com batata rústica e milkshake tradicional."
    }
    return recomendacoes.get(cluster_id, "🤔 Nenhuma recomendação disponível para este grupo.")

print(F"Sugestão personalizada de cardápio:")
print(recomendar_menu(predicted_cluster))