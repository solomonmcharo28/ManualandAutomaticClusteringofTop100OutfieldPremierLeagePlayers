import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# K-medians (L1 / Taxicab) clustering
# -----------------------------
def k_medians_l1(X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-6, random_state: int = 42):
    """
    K-medians clustering under the taxicab (L1/Manhattan) metric.

    - Assignment: nearest centroid by L1 distance
    - Update: centroid is coordinate-wise median (not mean!)
    """
    rng = np.random.default_rng(random_state)

    # Initialize centroids by sampling k points
    centroids = X[rng.choice(X.shape[0], size=k, replace=False)]

    for _ in range(max_iter):
        # Assign each point to closest centroid (L1)
        dists = np.sum(np.abs(X[:, None, :] - centroids[None, :, :]), axis=2)  # (n, k)
        labels = np.argmin(dists, axis=1)

        # Update centroids with coordinate-wise medians
        new_centroids = centroids.copy()
        for j in range(k):
            members = X[labels == j]
            if members.shape[0] == 0:
                # Empty cluster: re-seed to a random point
                new_centroids[j] = X[rng.integers(0, X.shape[0])]
            else:
                new_centroids[j] = np.median(members, axis=0)

        # Convergence check (L1 shift)
        shift = np.sum(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids


# -----------------------------
# 1. Load Excel file
# -----------------------------
file_path = "SerieABook.xlsx"
df = pd.read_excel(file_path)
OUTPUT_FILE = "SerieABook_with_clusters.xlsx"
# -----------------------------
# 2. Select required columns
# -----------------------------
cols = ["age", "weight_kg", "mcharo_fender_ln_price", "short_name"]
df = df[cols].dropna().reset_index(drop=True)
df_sub = (
    df[cols]
    .dropna()
    .reset_index(drop=False)   # keep original index!
)

# -----------------------------
# 3. Run L1 clustering (clusters decide color)
# -----------------------------
k = 4  # change this if you want more/fewer clusters

X_raw = df[["age", "weight_kg", "mcharo_fender_ln_price"]].to_numpy(dtype=float)

# Normalize so age/weight/price are comparable
mu = X_raw.mean(axis=0)
sigma = X_raw.std(axis=0)
sigma[sigma == 0] = 1.0
X = (X_raw - mu) / sigma

clusters, centroids = k_medians_l1(X, k=k)

# -----------------------------
# 4. Create 3D plot (cluster => color; short_name => label)
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
df_sub["cluster"] = clusters

df["cluster"] = np.nan
df.loc[df_sub["index"], "cluster"] = df_sub["cluster"].values

df.to_excel(OUTPUT_FILE, index=False)

sc = ax.scatter(
    df["age"],
    df["weight_kg"],
    df["mcharo_fender_ln_price"],
    c=clusters,          # clusters decide color
    cmap="tab10",
    alpha=0.65,
    s=35
)

# Labels are short_name
for i, row in df.iterrows():
    ax.text(
        row["age"],
        row["weight_kg"],
        row["mcharo_fender_ln_price"],
        str(row["short_name"]),  # short_name decides label
        fontsize=7
    )

ax.set_xlabel("Age")
ax.set_ylabel("Weight (kg)")
ax.set_zlabel("Mcharo-Fender ln(Price)")
ax.set_title("K-medians (Taxicab/L1) Clusters: Age vs Weight vs Mcharo-Fender ln(Price)")

# Add a colorbar showing cluster id
cbar = fig.colorbar(sc, ax=ax, pad=0.1, fraction=0.03)
cbar.set_label("Cluster")

plt.tight_layout()
plt.savefig("primerarecard.png", dpi=200)
plt.show()
