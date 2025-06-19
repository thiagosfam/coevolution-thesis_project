import pandas as pd
import matplotlib.pyplot as plt

# ───── CONFIG: map each archive label to its CSV file ─────
files = {
    "Vanilla HoF": "res_test_vs_random_spies.csv",
    "Custom Archive (size 50)": "res_test_vs_random_spies_ca_50.csv",
    "Custom Archive (size 100)": "res_test_vs_random_spies_ca_100.csv",
    "Composite Archive (Unbounded)": "res_test_vs_random_spies_composite_archive.csv",
    "Composite Archive (bounded)"  : "res_test_vs_random_spies_composite_archive50.csv"
}

for label, path in files.items():
    df = pd.read_csv(path)
    grand_mean = df["win_rate"].mean()
    print(f"{label:25s} -> Grand mean win rate: {grand_mean:.2%}")

plt.figure(figsize=(8, 6))

# Load each CSV, compute per-run average, and plot
for label, path in files.items():
    df = pd.read_csv(path)
    per_run = df.groupby("run")["win_rate"].mean()
    plt.plot(
        per_run.index,
        per_run.values,
        marker='o',
        linestyle='-',
        label=label
    )

plt.xlabel("Run")
plt.ylabel("Average Win Rate")
plt.title("Resistance Win Rate per Run for Different Archives")
plt.xticks(range(1, per_run.index.max() + 1))
plt.legend()
plt.tight_layout()
plt.show()
