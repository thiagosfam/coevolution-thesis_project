from pathlib import Path
import numpy as np
import pandas as pd
from engine.game import simulate_game
from evolutionary.archive_policy_final import ArchiveManager, Individual
import matplotlib.pyplot as plt

res_df = pd.read_csv("results/CustomArchivePerformance100/run_4/resistance_archive.csv")
spy_df = pd.read_csv("results/CustomArchivePerformance100/run_4/spy_archive.csv")

# Create a new archive manager
archive = ArchiveManager(max_size=30)

# Extract individuals from resistance DataFrame
res_gene_cols = [col for col in res_df.columns if col.startswith("gene_")]
for _, row in res_df.iterrows():
    genes = row[res_gene_cols].values.astype(np.float32)
    fitness = float(row["i_fitness"])
    generation = int(row["generation"])
    archive.add(Individual(genes=genes, fitness=fitness, role="resistance", generation=generation))

# Extract individuals from spy DataFrame
spy_gene_cols = [col for col in spy_df.columns if col.startswith("gene_")]
for _, row in spy_df.iterrows():
    genes = row[spy_gene_cols].values.astype(np.float32)
    fitness = float(row["i_fitness"])
    generation = int(row["generation"])
    archive.add(Individual(genes=genes, fitness=fitness, role="spy", generation=generation))

# Count archive sizes to confirm
print(len(archive.get_archive("resistance")))
print(len(archive.get_archive("spy")))

archive.cross_evaluate()
archive.compute_novelty(role='resistance', k=3)
archive.compute_novelty(role='spy', k=3)


print("\n First 5 Resistance in the Archive")
for ind in archive.get_archive(role='resistance')[:5]:
    print(ind)

print("\n First 5 Spies in the Archive")
for ind in archive.get_archive(role='spy')[:5]:
    print(ind)

# Now rerun pruning
res_pruned_A = archive.prune(method="multiobjective", role="resistance", alpha=1.0, beta=1.0)
spy_pruned_A = archive.prune(method="multiobjective", role="spy", alpha=1.0, beta=1.0)
res_pruned_B = archive.prune(method="clustering", role="resistance", k=4, keep_novel=2)
spy_pruned_B = archive.prune(method="clustering", role="spy", k=4, keep_novel=2)

# Summarize outputs
res_A_summary = [(ind.generation, round(ind.insertion_fitness, 2), round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in res_pruned_A]
spy_A_summary = [(ind.generation, round(ind.insertion_fitness, 2), round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in spy_pruned_A]
res_B_summary = [(ind.generation, round(ind.insertion_fitness, 2), round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in res_pruned_B]
spy_B_summary = [(ind.generation, round(ind.insertion_fitness, 2), round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in spy_pruned_B]

print("\n########## Summary Option A: Multiobjective (Fitness + Novelty) ##########")
print("\nResistance")
print(res_A_summary)
print("\nSpies")
print(spy_A_summary)

print("\n########## Summary Option B: Clustering + Novelty ##########")
print("\nResistance")
print(res_B_summary)
print("\nSpies")
print(spy_B_summary)

# Plotting function with highlights for Option A and B selections
def plot_novelty_vs_fitness_highlight(all_individuals, selected_A, selected_B, title):
    x_all = [ind.novelty_score for ind in all_individuals]
    y_all = [ind.archive_fitness for ind in all_individuals]

    x_A = [ind.novelty_score for ind in selected_A]
    y_A = [ind.archive_fitness for ind in selected_A]

    x_B = [ind.novelty_score for ind in selected_B]
    y_B = [ind.archive_fitness for ind in selected_B]

    plt.figure(figsize=(6, 5))
    plt.scatter(x_all, y_all, alpha=0.3, label="All")
    plt.scatter(x_A, y_A, color="blue", label="Option A (Fitness+Novelty)", marker="x")
    plt.scatter(x_B, y_B, color="red", label="Option B (Clustering)", marker="^")
    plt.xlabel("Novelty Score")
    plt.ylabel("Archive Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot for Resistance
plot_novelty_vs_fitness_highlight(
    archive.get_archive("resistance"),
    res_pruned_A,
    res_pruned_B,
    "Resistance Archive: Novelty vs Fitness (Pruned)"
)

# Plot for Spy
plot_novelty_vs_fitness_highlight(
    archive.get_archive("spy"),
    spy_pruned_A,
    spy_pruned_B,
    "Spy Archive: Novelty vs Fitness (Pruned)"
)