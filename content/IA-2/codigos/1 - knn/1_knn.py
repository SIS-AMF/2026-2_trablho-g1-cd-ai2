import math

# Dataset manual
jogos = [
    ("Skyrim", 100, 20, "RPG"),
    ("The Witcher", 120, 15, "RPG"),
    ("Doom", 10, 95, "FPS"),
    ("CS:GO", 5, 90, "FPS"),
    ("Mass Effect", 80, 40, "RPG"),
    ("Call of Duty", 12, 85, "FPS")
]

novo_jogo = ("Novo Game", 50, 50)

# Distância euclidiana
distancias = []
for nome, h, a, classe in jogos:
    d = math.sqrt((h - novo_jogo[1])**2 + (a - novo_jogo[2])**2)
    distancias.append((nome, classe, d))

# Ordenar por distância
distancias.sort(key=lambda x: x[2])

print("Vizinhos mais próximos:")
K = 3
for d in distancias[:K]: 
    print(d)

##### Contagem de classes

rpg = 0
fps = 0
for nome, classe, d in distancias[:K]:
    if classe == "RPG":
        rpg += 1
    else:
        fps += 1

# Decisão final
if rpg > fps:
    resultado = "RPG"
else:
    resultado = "FPS"

print(f"\nO novo jogo foi classificado como: {resultado}")

####### ---- Gráfico ----
import matplotlib.pyplot as plt

for nome, h, a, classe in jogos:
    cor = "blue" if classe == "RPG" else "red"
    plt.scatter(h, a, color=cor)

plt.scatter(novo_jogo[1], novo_jogo[2], color="green", marker="X", s=200)
plt.title(f"KNN (k={K}) → {resultado}")
plt.xlabel("Horas de História")
plt.ylabel("Intensidade da Ação")
plt.grid(True)
plt.show()

# ---- Normalização (Min-Max) ----
# Separar colunas
horas = [h for _, h, _, _ in jogos]
acao  = [a for _, _, a, _ in jogos]

min_h, max_h = min(horas), max(horas)
min_a, max_a = min(acao), max(acao)

def normalizar(x, minimo, maximo):
    return (x - minimo) / (maximo - minimo)

# Normalizar dataset
jogos_norm = []
for nome, h, a, classe in jogos:
    jogos_norm.append((nome, normalizar(h,min_h,max_h), normalizar(a,min_a,max_a), classe))

novo_norm = (novo_jogo[0],
             normalizar(novo_jogo[1],min_h,max_h),
             normalizar(novo_jogo[2],min_a,max_a))

# ---- Calcular distâncias ----
distancias = []
for nome, h, a, classe in jogos_norm:
    d = math.sqrt((h - novo_norm[1])**2 + (a - novo_norm[2])**2)
    distancias.append((nome, classe, d))

distancias.sort(key=lambda x: x[2])
vizinhos = distancias[:K]

# Contagem
rpg = sum(1 for _, c, _ in vizinhos if c=="RPG")
fps = K - rpg
resultado = "RPG" if rpg > fps else "FPS"

print("Novo jogo classificado como:", resultado)