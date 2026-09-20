# MovieLens 100k — Fast user-KNN with Euclidean distance in SVD space
# - Loads local u.data / u.item
# - Train/test split by interactions
# - Dimensionality reduction (TruncatedSVD) for speed
# - NearestNeighbors(metric="euclidean") on user factors
# - Inverse-distance weighted prediction with safe fallbacks
# - Reports RMSE and shows a sample prediction

import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# -------------------------
# 1) Load local MovieLens
# -------------------------
u_data_path = "./ml-100k/u.data"
u_item_path = "./ml-100k/u.item"

if not (os.path.exists(u_data_path) and os.path.exists(u_item_path)):
    raise FileNotFoundError("Check that ./ml-100k/u.data and ./ml-100k/u.item exist.")

# u.data is tab-separated: user_id, item_id, rating, timestamp
df = pd.read_csv(u_data_path, sep="\t", header=None, names=["user", "item", "rating", "ts"])
df["user"] = df["user"].astype(int)
df["item"] = df["item"].astype(int)
df["rating"] = df["rating"].astype(float)

# u.item is pipe-separated; first fields are: item_id | title | ...
items_df = pd.read_csv(u_item_path, sep="|", header=None, encoding="latin-1")
items_df = items_df[[0, 1]].rename(columns={0: "item", 1: "title"})
items_df["item"] = items_df["item"].astype(int)

# -------------------------
# 2) Index maps & split
# -------------------------
users = np.sort(df["user"].unique())
items = np.sort(df["item"].unique())
u2ix = {u:i for i,u in enumerate(users)}
i2ix = {m:i for i,m in enumerate(items)}

df["uix"] = df["user"].map(u2ix)
df["iix"] = df["item"].map(i2ix)

# Hold out 25% interactions at random (stratified by user if you prefer)
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

num_users = len(users)
num_items = len(items)

# -------------------------
# 3) Build sparse train matrix
# -------------------------
R_train = sparse.csr_matrix(
    (train_df["rating"].values, (train_df["uix"].values, train_df["iix"].values)),
    shape=(num_users, num_items)
)

# Useful caches
# Per-user dict of item->rating (train only)
user_ratings = [{} for _ in range(num_users)]
for r in zip(train_df["uix"], train_df["iix"], train_df["rating"]):
    user_ratings[r[0]][r[1]] = r[2]

# Baselines
global_mean = train_df["rating"].mean()
# Per-user mean (fallback)
user_mean = np.full(num_users, global_mean, dtype=np.float64)
for u in range(num_users):
    vals = list(user_ratings[u].values())
    if vals:
        user_mean[u] = float(np.mean(vals))
# Per-item mean (fallback)
item_sum = np.zeros(num_items, dtype=np.float64)
item_cnt = np.zeros(num_items, dtype=np.int32)
for _, row in train_df.iterrows():
    iix = int(row["iix"])          # <-- cast!
    r = float(row["rating"])
    item_sum[iix] += r
    item_cnt[iix] += 1
item_mean = np.where(item_cnt > 0, item_sum / item_cnt, global_mean)

# -------------------------
# 4) Dimensionality reduction for speed
#    (Euclidean works well in low dims; trees get effective)
# -------------------------
latent_dim = 64  # 32–128 works well; adjust if you like
svd = TruncatedSVD(n_components=latent_dim, random_state=42)
U_factors = svd.fit_transform(R_train)  # shape: (num_users, latent_dim)

# Optional: mean/variance standardization helps Euclidean sometimes
# We'll center to zero-mean per dimension:
U_factors = U_factors - U_factors.mean(axis=0, keepdims=True)

# -------------------------
# 5) Pre-fit neighbors (Euclidean, kd_tree/ball_tree auto-chosen)
# -------------------------
k_neighbors = 20  # typical 20–100; more neighbors = slower but more robust
nn = NearestNeighbors(
    n_neighbors=k_neighbors,
    metric="euclidean",
    algorithm="auto",  # KD-tree/Ball-tree chosen under the hood
    n_jobs=-1
)
nn.fit(U_factors)

# For all users, cache their neighbor indices & distances (vectorized, fast)
dists_all, nbrs_all = nn.kneighbors(U_factors, return_distance=True)

# -------------------------
# 6) Prediction function
# -------------------------
def predict_rating(uix: int, iix: int, eps: float = 1e-6) -> float:
    """
    Predict rating for (user uix, item iix) using inverse-distance weighted
    neighbor average over users who rated the item in the TRAIN set.
    Fallback: blend of user_mean and item_mean -> then global_mean.
    """
    nbrs = nbrs_all[uix]
    dists = dists_all[uix]

    # Collect neighbor ratings for the target item
    numer = 0.0
    denom = 0.0
    for d, v in zip(dists, nbrs):
        r = user_ratings[v].get(iix, None)
        if r is not None:
            w = 1.0 / (d + eps)  # inverse-distance weight
            numer += w * r
            denom += w

    if denom > 0:
        return numer / denom

    # Fallbacks if no neighbor rated the item:
    # 1) simple blend of user and item means
    um = user_mean[uix]
    im = item_mean[iix]
    backoff = 0.6 * um + 0.4 * im

    # Guard against NaNs (shouldn't occur, but just in case)
    if not np.isfinite(backoff):
        backoff = global_mean
    return backoff

# -------------------------
# 7) Evaluate on test
# -------------------------
y_true = test_df["rating"].values
y_pred = np.empty_like(y_true, dtype=np.float64)

for idx, row in enumerate(test_df.itertuples(index=False)):
    # row: (user, item, rating, ts, uix, iix)
    y_pred[idx] = predict_rating(row.uix, row.iix)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# -------------------------
# 8) Show one example prediction
# -------------------------
sample_uid = users[0]          # raw user id
sample_iid = items[0]          # raw item id
sample_uix = u2ix[sample_uid]
sample_iix = i2ix[sample_iid]
est = predict_rating(sample_uix, sample_iix)

# Map to title if available
title = items_df.loc[items_df["item"] == sample_iid, "title"]
title = title.values[0] if len(title) else f"item {sample_iid}"

print(f"Predicted rating for user {sample_uid} on '{title}' [{sample_iid}]: {est:.2f}")
