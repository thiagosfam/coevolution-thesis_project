import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ====== CONFIG ======
SPY_PATTERN = "results/CompositeArchive/run_*/spy_stats.csv"
RES_PATTERN = "results/CompositeArchive/run_*/resistance_stats.csv"

# ====== LOAD RUNS WITH AUTO-TAGGING ======
def load_all_runs(file_pattern):
    all_dfs = []
    for path in sorted(glob.glob(file_pattern)):
        df = pd.read_csv(path)
        run_id = int(os.path.basename(os.path.dirname(path)).split("_")[1])  # from run_#
        df['run_id'] = run_id
        df['generation'] = df.index
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

# ====== PROCESS GROUPS ======
def process_group(df):
    avg = df.groupby('generation').agg({
        'avg_fitness': 'mean',
        'max_fitness': 'mean',
        'std_fitness': 'mean'
    }).reset_index()
    std = df.groupby('generation')['avg_fitness'].std().reset_index(name='std_dev')
    return avg.merge(std, on='generation')

# ====== LOAD & PROCESS ======
spy_df = load_all_runs(SPY_PATTERN)
res_df = load_all_runs(RES_PATTERN)
spy_avg = process_group(spy_df)
res_avg = process_group(res_df)

# ====== PLOT ======
plt.figure(figsize=(12, 6))

# ---- Spies ----
plt.plot(spy_avg['generation'], spy_avg['avg_fitness'],
         label='Spies - Avg Fitness', color='red', linewidth=1.5)
plt.plot(spy_avg['generation'], spy_avg['max_fitness'],
         label='Spies - Max Fitness', color='red', linewidth=1.0, linestyle='-', alpha=0.5)
plt.fill_between(spy_avg['generation'],
                 spy_avg['avg_fitness'] - spy_avg['std_dev'],
                 spy_avg['avg_fitness'] + spy_avg['std_dev'],
                 color='red', alpha=0.2)

# ---- Resistance ----
plt.plot(res_avg['generation'], res_avg['avg_fitness'],
         label='Resistance - Avg Fitness', color='blue', linewidth=1.5)
plt.plot(res_avg['generation'], res_avg['max_fitness'],
         label='Resistance - Max Fitness', color='blue', linewidth=1.0, linestyle='-', alpha=0.5)
plt.fill_between(res_avg['generation'],
                 res_avg['avg_fitness'] - res_avg['std_dev'],
                 res_avg['avg_fitness'] + res_avg['std_dev'],
                 color='blue', alpha=0.2)

# Setting Y-ticks
# Force y‐range 0.0 → 1.0
plt.ylim(0.0, 1.0)
# Set ticks every 10%
plt.yticks(np.arange(0.0, 1.01, 0.1))

# ====== FINAL TOUCHES ======
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("HoF Archive Average & Max Fitness Across 10 Runs")
plt.legend(loc="lower right")
#plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hof_avg_fitness_over_10_runs.png", dpi=300)
plt.show()
