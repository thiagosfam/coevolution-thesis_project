import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize

BASE_DIR = "results"
RUNS     = range(1, 11)
ROLES = {
    "resistance": {"genes": 14, "label": "r"},
    "spy":        {"genes": 10, "label": "s"}
}

for role, params in ROLES.items():
    G     = params["genes"]
    label = params["label"]

    # ─── #2 POOLED‐ELITE PCA ──────────────────────────────────────────────────────
    records = []
    for run in RUNS:
        fp = os.path.join(BASE_DIR,
                          f"sequential/run_{run}/sequential_coevolution_{role}_ancestral_line.csv")
        df = pd.read_csv(fp)
        for _, row in df.iterrows():
            gen   = int(row["generation"])
            genes = [row[f"gene_{i}"] for i in range(G)]
            records.append([run, gen] + genes)

    cols  = ["run","generation"] + [f"gene_{i}" for i in range(G)]
    df_all = pd.DataFrame(records, columns=cols)

    X_all = df_all[[f"gene_{i}" for i in range(G)]].values
    gens_all = df_all["generation"].values


    pca_all = PCA(n_components=G)
    Xp_all  = pca_all.fit_transform(X_all)

    # Scree
    plt.figure(figsize=(5,4))
    evr = pca_all.explained_variance_ratio_
    plt.plot(np.arange(1, G+1), evr, "o-")
    plt.title(f"{role.title()} Scree (pooled elites)")
    plt.xlabel("PC #")
    plt.ylabel("Explained var ratio")
    plt.xticks(np.arange(1, G+1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{role}_pooled_elite_scree.png", dpi=300)
    plt.show()

    # Scatter PC1 vs PC2, colored by generation
    norm = Normalize(vmin=1, vmax=X_all.shape[0]//len(RUNS))
    plt.figure(figsize=(6,5))
    sc = plt.scatter(
        Xp_all[:,0], Xp_all[:,1],
        c=gens_all, cmap="RdYlGn_r", norm=norm,
        s=30, edgecolor="k", linewidth=0.2
    )
    cbar = plt.colorbar(sc, pad=0.01)
    cbar.set_label("Generation")
    plt.title(f"{role.title()} PCA Scatter (pooled elites)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{role}_pooled_elite_scatter.png", dpi=300)
    plt.show()

    # Loadings for PC1 & PC2
    load1 = pca_all.components_[0]
    load2 = pca_all.components_[1]
    x = np.arange(G)
    plt.figure(figsize=(6,3))
    plt.bar(x-0.2, load1, width=0.4, label="PC1")
    plt.bar(x+0.2, load2, width=0.4, label="PC2", alpha=0.7)
    plt.xticks(x, [f"{label}{i+1}" for i in x], rotation=45)
    plt.ylabel("Loading")
    plt.title(f"{role.title()} Loadings (pooled elites)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{role}_pooled_elite_loadings.png", dpi=300)
    plt.show()


    # ─── #3 MEAN‐ELITE PCA ────────────────────────────────────────────────────────
    # build one DataFrame per run, indexed by generation
    dfs = []
    for run in RUNS:
        fp = os.path.join(BASE_DIR,
                          f"sequential/run_{run}/sequential_coevolution_{role}_ancestral_line.csv")
        dfi = pd.read_csv(fp)[["generation"] + [f"gene_{i}" for i in range(G)]]
        dfi = dfi.set_index("generation")
        dfs.append(dfi)

    # concatenate side‑by‑side and average across runs
    concat = pd.concat(dfs, axis=1, keys=RUNS)
    # extract mean per generation
    mean_df = pd.DataFrame(index=dfs[0].index)
    for i in range(G):
        # get all columns named gene_i across the 10 runs
        mean_df[f"gene_{i}"] = concat.xs(f"gene_{i}", axis=1, level=1).mean(axis=1)

    X_mean = mean_df.values
    gens_mean = mean_df.index.values

    N_GEN = len(gens_mean)

    pca_mean = PCA(n_components=G)
    Xp_mean  = pca_mean.fit_transform(X_mean)

    # Scree (mean‐elite)
    plt.figure(figsize=(5,4))
    evr2 = pca_mean.explained_variance_ratio_
    plt.plot(np.arange(1, G+1), evr2, "o-")
    plt.title(f"{role.title()} Scree (mean elites)")
    plt.xlabel("PC #")
    plt.ylabel("Explained var ratio")
    plt.xticks(np.arange(1, G+1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{role}_mean_elite_scree.png", dpi=300)
    plt.show()

    # Scatter PC1 vs PC2 of the mean elites
    plt.figure(figsize=(6,5))
    sc2 = plt.scatter(
        Xp_mean[:,0], Xp_mean[:,1],
        c=gens_mean, cmap="RdYlGn_r", norm=Normalize(vmin=1, vmax=N_GEN),
        s=40, edgecolor="k", linewidth=0.3
    )
    cbar2 = plt.colorbar(sc2, pad=0.01)
    cbar2.set_label("Generation")
    plt.title(f"{role.title()} PCA Scatter (mean elites)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{role}_mean_elite_scatter.png", dpi=300)
    plt.show()

    # Loadings (mean‐elite)
    lm1 = pca_mean.components_[0]
    lm2 = pca_mean.components_[1]
    plt.figure(figsize=(6,3))
    plt.bar(x-0.2, lm1, width=0.4, label="PC1")
    plt.bar(x+0.2, lm2, width=0.4, label="PC2", alpha=0.7)
    plt.xticks(x, [f"{label}{i+1}" for i in x], rotation=45)
    plt.ylabel("Loading")
    plt.title(f"{role.title()} Loadings (mean elites)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{role}_mean_elite_loadings.png", dpi=300)
    plt.show()
