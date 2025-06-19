import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Resistance Heatmap (r1–r14) =====
df_res = pd.read_csv("results/diverse_HoF/run_10/coevolution_resistance_ancestral_line_diverse_HoF.csv")
res_genes = [f"gene_{i}" for i in range(14)]
res_matrix = df_res[res_genes].T.values
res_labels = [f"r{i+1}" for i in range(14)]

plt.figure(figsize=(12, 4))
sns.heatmap(
    res_matrix,
    cmap="RdBu_r",
    center=0,
    yticklabels=res_labels,
    xticklabels=20
)
plt.xlabel("Generation")
plt.title("Resistance Elite Gene Values (r10-r14) Over Time")
plt.tight_layout()
plt.savefig("resistance_run10_diverse_heatmap.png", dpi=300)
plt.show()


# ===== Spy Heatmap (s1–s10) =====
# Load
df_spy = pd.read_csv("results/diverse_HoF/run_10/coevolution_spy_ancestral_line_diverse_HoF.csv")
spy_genes = [f"gene_{i}" for i in range(10)]
spy_matrix = df_spy[spy_genes].T.values
spy_labels = [f"s{i+1}" for i in range(10)]

# Plot
plt.figure(figsize=(12, 3))
ax = sns.heatmap(
    spy_matrix,
    cmap="RdBu_r",
    center=0,
    yticklabels=spy_labels,
    xticklabels=20
)
ax.set_xlabel("Generation")
ax.set_title("Spy Elite Gene Values (s1-s10) Over Time")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')

plt.tight_layout()
plt.savefig("spies_run10_diverse_heatmap.png", dpi=300)
plt.show()
