import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from pathlib import Path

# === 1) carregar imagem ===
IMG_PATH = "sua_imagem.jpeg"  # troque pelo arquivo do grupo
assert Path(IMG_PATH).exists(), "Coloque uma imagem no diretório e ajuste IMG_PATH."

img = Image.open(IMG_PATH).convert("RGB")
# redimensiona p/ agilizar (opcional)
max_side = 512
ratio = max(img.size) / max_side if max(img.size) > max_side else 1
img_small = img.resize((int(img.width/ratio), int(img.height/ratio)))

arr = np.array(img_small)          # (H, W, 3)
pixels = arr.reshape(-1, 3)        # (N, 3) no espaço RGB

# === 2) escolher K e treinar K-means ===
K = 30  # experimente 3, 5, 8, 12
kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
labels = kmeans.fit_predict(pixels)
centroids = kmeans.cluster_centers_.astype(np.uint8)  # cores da paleta

# === 3) reconstruir imagem com paleta (centróides) ===
quantized = centroids[labels].reshape(arr.shape)

# === 4) mostrar original, quantizada e paleta ===
def show_palette(colors):
    swatch = np.zeros((50, 60*len(colors), 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        swatch[:, 60*i:60*(i+1), :] = c
    return swatch

plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.imshow(arr); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(quantized); plt.title(f"Quantizada (K={K})"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(show_palette(centroids)); plt.title("Paleta"); plt.axis("off")
plt.tight_layout(); plt.show()

# === 5) método do cotovelo (opcional) ===
Ks = [2,3,4,5,6,8,10,12]
inertias = []
for k in Ks:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(pixels)
    inertias.append(km.inertia_)
plt.figure(figsize=(5,4))
plt.plot(Ks, inertias, marker="o")
plt.xlabel("K"); plt.ylabel("Inertia (SSE)")
plt.title("Método do cotovelo")
plt.show()
