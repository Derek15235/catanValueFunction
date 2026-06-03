"""pca_analysis.py — PCA + k-means on Catan game state snapshots.

Produces three plots in results/pca/:
  1. pca_by_phase.png: PCA scatter colored by VP phase
  2. pca_kmeans_winrate.png: PCA scatter colored by k-means cluster win rate
  3. pca_cluster_phases.png: bar chart phase composition of each cluster

Usage:
    uv run python pca_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from schema import FEATURE_ORDERING

OUT_DIR = Path("results/pca")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 6
SAMPLE_SIZE = 20_000   


df = pd.read_parquet("data/snapshots.parquet")
df = df.sample(n=SAMPLE_SIZE, random_state=0).reset_index(drop=True)

X = df[FEATURE_ORDERING].values  
y = df["label"].values           
max_vp = df["max_vp"].values      

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")


def vp_to_bucket(vp):
    if vp < 4:  return "2-4"
    if vp < 6:  return "4-6"
    if vp < 8:  return "6-8"
    if vp < 10: return "8-10"
    if vp < 12: return "10-12"
    return "12-15"

buckets = np.array([vp_to_bucket(v) for v in max_vp])
bucket_order = ["2-4", "4-6", "6-8", "8-10", "10-12", "12-15"]


plt.figure(figsize=(8, 6))
for b in bucket_order:
    mask = buckets == b
    plt.scatter(coords[mask, 0], coords[mask, 1], label=b, alpha=0.3, s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Game states colored by VP phase")
plt.legend(title="max VP")
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_by_phase.png", dpi=150)
plt.close()
print("Saved pca_by_phase.png")


km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
clusters = km.fit_predict(coords)

win_rates = np.zeros(N_CLUSTERS)
for c in range(N_CLUSTERS):
    in_cluster = clusters == c
    win_rates[c] = y[in_cluster].mean()
    print(f"Cluster {c}: {in_cluster.sum():>5} snapshots, win rate = {win_rates[c]:.2f}")


plt.figure(figsize=(8, 6))
point_winrate = win_rates[clusters]   
plt.scatter(coords[:, 0], coords[:, 1], c=point_winrate, cmap="RdYlGn", alpha=0.3, s=5, vmin=0, vmax=1)
plt.colorbar(label="cluster win rate")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-means clusters colored by win rate")
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_kmeans_winrate.png", dpi=150)
plt.close()
print("Saved pca_kmeans_winrate.png")


composition = pd.crosstab(clusters, buckets, normalize="index")
composition = composition[bucket_order]

print()
print("Phase composition by cluster (rows sum to 1.0):")
print(composition.to_string(float_format="%.3f"))

import json
metrics = {
    "pc1_variance": float(pca.explained_variance_ratio_[0]),
    "pc2_variance": float(pca.explained_variance_ratio_[1]),
    "n_clusters": N_CLUSTERS,
    "sample_size": SAMPLE_SIZE,
    "clusters": [
        {
            "id": int(c),
            "n": int((clusters == c).sum()),
            "win_rate": float(win_rates[c]),
            "phase_composition": {b: float(composition.iloc[c][b]) for b in bucket_order},
        }
        for c in range(N_CLUSTERS)
    ],
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("\nSaved metrics.json")

composition.index = [f"c{c} (wr={win_rates[c]:.2f})" for c in composition.index]

composition.plot(kind="bar", figsize=(9, 5), colormap="viridis", width=0.8)
plt.xlabel("Cluster")
plt.ylabel("Fraction of snapshots")
plt.title("VP phase composition of each cluster")
plt.legend(title="max VP", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_cluster_phases.png", dpi=150)
plt.close()
print("Saved pca_cluster_phases.png")