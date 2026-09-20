import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Dataset realista
clients = pd.DataFrame({
    'Client': ['Ana', 'Bruno', 'Carlos', 'Daniela', 'Eduardo', 'Fernanda', 
               'Gustavo', 'Helena', 'Igor', 'Juliana', 'Kleber', 'Luiza'],
    'AnnualSpend': [9000, 1200, 3000, 7000, 2000, 8500, 4000, 9500, 2200, 7800, 2500, 8900],
    'Frequency': [8, 1, 3, 7, 2, 9, 4, 10, 2, 8, 2, 9],
    'Recency': [5, 120, 60, 10, 90, 3, 30, 2, 80, 5, 70, 3]
})

# Selecionar features
X = clients[['AnnualSpend', 'Frequency', 'Recency']]

# Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
clients['Cluster'] = kmeans.fit_predict(X_scaled)

# 🎨 Gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Separar os dados por cluster
colors = ['red', 'green', 'blue']
for cluster in clients['Cluster'].unique():
    subset = clients[clients['Cluster'] == cluster]
    ax.scatter(
        subset['AnnualSpend'], 
        subset['Frequency'], 
        subset['Recency'], 
        color=colors[cluster], 
        label=f'Cluster {cluster}',
        s=60
    )

# Anotar os nomes dos clientes
for i, row in clients.iterrows():
    ax.text(row['AnnualSpend'], row['Frequency'], row['Recency'], row['Client'], size=8)

# Rótulos e título
ax.set_title("Customer Segmentation (K-Means) - 3D View")
ax.set_xlabel("Annual Spend (R$)")
ax.set_ylabel("Frequency (Orders/Month)")
ax.set_zlabel("Recency (days since last purchase)")
ax.legend()
plt.show()
