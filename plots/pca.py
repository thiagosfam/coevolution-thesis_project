import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize

# ───── CONFIG ─────
BASE_DIR = "results"
ROLES = {
    "resistance": 14,  # number of genes
    "spy":        10
}

for role, gene_count in ROLES.items():
    # 1) Find the ancestral-line CSV (we just pick the first run here)
    pattern = os.path.join(BASE_DIR, "run_*", f"coevolution_resistance_hall_of_fame_HoF.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ancestral‐line file found for role '{role}' with pattern {pattern}")
    path = files[0]
    print(f"\nLoading {role} data from: {path}")

    # 2) Load and extract the gene matrix (gens × gene_count)
    df = pd.read_csv(path)
    gene_cols = [f"gene_{i}" for i in range(gene_count)]
    X = df[gene_cols].values  # shape (n_generations, gene_count)
    n_gens = X.shape[0]

    # 3) Fit full PCA
    pca = PCA(n_components=gene_count)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    # ─── 4) Scree Plot ───
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, gene_count+1), evr, marker='o', linestyle='-')
    plt.title(f"Scree Plot — {role.title()} PCA")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(np.arange(1, gene_count+1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ─── 5) PC1 vs PC2 Scatter (green→red by generation) ───
    generations = np.arange(1, n_gens + 1)
    norm = Normalize(vmin=1, vmax=n_gens)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=generations,
        cmap="RdYlGn_r",
        norm=norm,
        s=50,
        edgecolor='k', linewidth=0.3
    )
    cbar = plt.colorbar(sc, pad=0.01)
    cbar.set_label("Generation")
    plt.title(f"PCA Scatter — {role.title()} Elite (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
