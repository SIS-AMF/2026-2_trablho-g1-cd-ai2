import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# =======================
# 1. Criando o dataset
# =======================
np.random.seed(42)

# Cada linha: [velocidade (km/h), horário (24h)]
X = np.array([
    [48, 22.1],
    [52, 21.9],
    [50, 22.3],
    [47, 22.0],
    [51, 22.2],
    [49, 21.8],
    [39, 23.4],
    [41, 23.6],
    [38, 23.5],
    [42, 23.3],
    [40, 23.5],
    [37, 23.7],
    [108, 1.1],
    [112, 1.3],
    [115, 1.2],
    [109, 1.0],
    [111, 1.2],
    [80, 2.0],  
    [55, 4.5]
])

# veículos normais (média 50 km/h, às 22h)
# a = np.column_stack([np.random.normal(50, 5, 30), np.random.normal(22, 0.3, 30)])
# veículos normais (média 40 km/h, às 23h30)
# b = np.column_stack([np.random.normal(40, 4, 25), np.random.normal(23.5, 0.2, 25) ])
# alta velocidade, madrugada
# c = np.column_stack([np.random.normal(110, 8, 10), np.random.normal(1.2, 0.05, 10)])
# Outliers isolados
# d = np.array([[20, 20.0], [40, 2.0], [55, 4.5]])
# Dataset completo
# X = np.vstack([a, b, c, d])

# =======================
# 2. Normalização
# =======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =======================
# 3. DBSCAN
# =======================
dbscan = DBSCAN(eps=0.7, min_samples=4)
labels = dbscan.fit_predict(X_scaled)

# =======================
# 4. Visualização
# =======================
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=70, edgecolors='k')
plt.xlabel("Velocidade (km/h)")
plt.ylabel("Horário (h)")
plt.title("Clusters detectados pelo DBSCAN")
plt.grid(True)
plt.show()
