import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# CONFIG
BASE = "results"
ROLE = "spy"   # or "resistance"
GENE_COUNT = 10 if ROLE=="spy" else 14
LABEL = "s" if ROLE=="spy" else "r"

# 1. Load all runs’ ancestral lines into a 3D array: (runs, genes, generations)
runs = sorted(glob.glob(f"{BASE}/run_*/coevolution_{ROLE}_ancestral_line_HoF.csv"))
all_data = np.zeros((len(runs), GENE_COUNT, 200))  # adjust 200 if needed

for idx, path in enumerate(runs):
    df = pd.read_csv(path)
    genes = [f"gene_{i}" for i in range(GENE_COUNT)]
    # DataFrame is (200 gens × GENE_COUNT); transpose to (GENE_COUNT × 200)
    all_data[idx] = df[genes].T.values

# 2. Compute mean & std across the first axis (runs)
mean_mat = all_data.mean(axis=0)
std_mat  = all_data.std(axis=0)

n_gens = mean_mat.shape[1]
xtick_step = 20
xticks = np.arange(0, n_gens, xtick_step)
xtick_labels = (xticks + 1).astype(int)  # shift from 0-index to generation number


yticks = [f"{LABEL}{i+1}" for i in range(GENE_COUNT)]

# --- plot mean heatmap ---
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(mean_mat,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            yticklabels=yticks,
            cbar_kws={'label': 'Mean gene value'})

# set only every 20th generation as an x-tick
ax.set_xticks(xticks + 0.5)           # +0.5 centers the tick on the cell
ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
ax.set_xlabel("Generation")
ax.set_title(f"{ROLE.title()} Elite Mean Gene Values")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')

plt.tight_layout()
plt.savefig(f"{ROLE.title()}_heatmap_avg.png", dpi=300)
plt.show()

# --- plot std heatmap ---
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(std_mat,
            ax=ax,
            cmap="magma",
            yticklabels=yticks,
            cbar_kws={'label': 'Std Dev of gene value'})

ax.set_xticks(xticks + 0.5)
ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
ax.set_xlabel("Generation")
ax.set_title(f"{ROLE.title()} Elite Gene Std Dev")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')

plt.tight_layout()
plt.savefig(f"{ROLE.title()}_heatmap_std.png", dpi=300)
plt.show()