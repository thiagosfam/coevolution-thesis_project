import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

summary = []

for i in range(1, 11):  # assuming 10 runs
    run_path = f"results/CompositeArchive/run_{i}"
    res_df = pd.read_csv(f"{run_path}/resistance_archive.csv")
    spy_df = pd.read_csv(f"{run_path}/spy_archive.csv")
    coverage_df = pd.read_csv(f"{run_path}/coverage_over_time.csv")

    summary.append({
        "run": i,
        "res_size": len(res_df),
        "spy_size": len(spy_df),
        "res_coverage": coverage_df["res_coverage_pct"].iloc[-1],
        "spy_coverage": coverage_df["spy_coverage_pct"].iloc[-1],
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/final_summary.csv", index=False)

# Load the summary CSV
df = pd.read_csv("results/final_summary.csv")

runs = df["run"].astype(str)
res_sizes = df["res_size"]
spy_sizes = df["spy_size"]
res_cov = df["res_coverage"].round(2)
spy_cov = df["spy_coverage"].round(2)

x = np.arange(len(runs))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))

# Plot full bars
res_bars = ax.bar(x - width/2, res_sizes, width, label='Resistance', color='blue')
spy_bars = ax.bar(x + width/2, spy_sizes, width, label='Spy', color='red')

# Overlay bars for coverage
res_cov_bars = ax.bar(x - width/2, [r * c / 100 for r, c in zip(res_sizes, res_cov)],
                      width, hatch='///', alpha=0.4, color='white', edgecolor='black')

spy_cov_bars = ax.bar(x + width/2, [s * c / 100 for s, c in zip(spy_sizes, spy_cov)],
                      width, hatch='\\\\\\', alpha=0.4, color='white', edgecolor='black')

# Add text labels
for i in range(len(runs)):
    ax.text(x[i] - width/2, res_sizes[i] + 0.5, f'{res_sizes[i]}', ha='center', fontsize=8)
    ax.text(x[i] + width/2, spy_sizes[i] + 0.5, f'{spy_sizes[i]}', ha='center', fontsize=8)
    ax.text(x[i] - width/2, res_sizes[i] * res_cov[i] / 100 / 2, f'{res_cov[i]}%', ha='center', va='center', fontsize=8, color='black')
    ax.text(x[i] + width/2, spy_sizes[i] * spy_cov[i] / 100 / 2, f'{spy_cov[i]}%', ha='center', va='center', fontsize=8, color='black')

ax.set_ylabel('Final Archive Size')
ax.set_title('Final Archive Sizes and Coverage % per Run')
ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.legend()

plt.tight_layout()
plt.show()