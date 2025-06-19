import random
import numpy as np
import pandas as pd
from pathlib import Path
from engine.game import simulate_game
from numpy.random import default_rng

# ───── CONFIG ─────
N_RUNS        = 10
N_TEST_SPIES  = 1000
N_GAMES       = 100
SEED          = 1234
GENE_RANGE    = (0, 1)  # spy genes range
ARCHIVE_ROOT = Path("results/MultiObjectiveArchive")
rng = default_rng(SEED)

# Pre-generate 1 000 random spy genomes (each is a length-10 list)
test_spies = rng.uniform(GENE_RANGE[0], GENE_RANGE[1],
                         size=(N_TEST_SPIES, 10)).tolist()

results = []

for run in range(1, N_RUNS + 1):
    # Load the resistance archive for this run
    res_csv = ARCHIVE_ROOT / f"run_{run}" / "resistance_archive.csv"
    df_res = pd.read_csv(res_csv)
    gene_cols = [c for c in df_res.columns if c.startswith("gene_")]
    resistance_pop = df_res[gene_cols].values.tolist()

    print(f"Run {run}: testing {len(resistance_pop)} resistance × "
          f"{N_TEST_SPIES} random spies")

    for idx, res_genes in enumerate(resistance_pop):
        wins = 0
        total_games = 0

        # play each resistance vs each random spy
        for spy_genes in test_spies:
            # simulate N_GAMES for this pairing
            players = [res_genes]*3 + [spy_genes]*2
            for _ in range(N_GAMES):
                winner = simulate_game(list_of_players=players, verbose=False)
                if winner == "resistance":
                    wins += 1
                total_games += 1

        # compute win-rate over all pairings
        win_rate = wins / total_games
        results.append({
            "run": run,
            "res_index": idx,
            "win_rate": win_rate
        })

# Save and summarise
df_out = pd.DataFrame(results)
df_out.to_csv("res_test_vs_random_spies_MOA.csv", index=False)

summary = df_out.groupby("run")["win_rate"].agg(["mean", "std"])
print("\nPer-run Resistance test vs random spies:\n", summary)
