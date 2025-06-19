# pca_grid_singlecolour_v2.py
import re, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

BASE_DIR   = Path("results/HoF_archive")
RUN_PATTERN = "run_*"
NROWS, NCOLS = 2, 5

ROLES = {
    "resistance": {"genes":14, "file":"resistance_archive.csv", "color":"blue"},
    "spy":        {"genes":10, "file":"spy_archive.csv",        "color":"red"},
}

def natural_sort(paths):
    return sorted(paths, key=lambda p: int(re.search(r"run[_\-]?(\d+)", p.as_posix()).group(1)))

for role, cfg in ROLES.items():
    gene_cols = [f"gene_{i}" for i in range(cfg["genes"])]
    csv_paths = natural_sort(
        BASE_DIR.glob(f"{RUN_PATTERN}/{cfg['file']}"))

    if not csv_paths:
        print(f"No files named {cfg['file']} for {role}; skipping.")
        continue

    pcs_list, xlims, ylims = [], [], []

    for csv in csv_paths:
        df = pd.read_csv(csv)
        X  = df[gene_cols].to_numpy(float)
        X  = (X - X.mean(0)) / (X.std(0) + 1e-9)
        pcs = PCA(n_components=2).fit_transform(X)
        pcs_list.append(pcs)
        xlims.extend([pcs[:,0].min(), pcs[:,0].max()])
        ylims.extend([pcs[:,1].min(), pcs[:,1].max()])

    pad_x = 0.05 * (max(xlims)-min(xlims))
    pad_y = 0.05 * (max(ylims)-min(ylims))
    xlim, ylim = (min(xlims)-pad_x, max(xlims)+pad_x), (min(ylims)-pad_y, max(ylims)+pad_y)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(3*NCOLS, 3*NROWS),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, csv, pcs in zip(axes, csv_paths, pcs_list):
        ax.scatter(pcs[:,0], pcs[:,1], color=cfg["color"],
                   s=20, alpha=0.85, edgecolor="k", lw=.25)
        ax.set_title(csv.parent.name, fontsize=9, pad=3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)

    for ax in axes[len(csv_paths):]:
        ax.axis("off")

    # Shared X/Y labels
    fig.text(0.5, 0.04, "PC 1", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "PC 2", va="center", rotation="vertical", fontsize=12)

    fig.suptitle(f"PCA HoF Archive - {role.title()} Final Archives (10 runs)",
                 y=0.98, fontsize=15)

    # single legend
    fig.legend([plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=cfg["color"], markeredgecolor='k', lw=0)],
               [role.title()], loc="upper right", frameon=False)

    plt.tight_layout(rect=[0.05,0.05,0.95,0.94])

    out = BASE_DIR / f"pca_grid_{role}.png"
    plt.savefig(out, dpi=150)
    print("✔ saved", out)
