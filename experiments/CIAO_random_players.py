import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from engine.game import simulate_game 

def load_ciao_data(base_path="results", num_runs=10):
    spy_runs = []
    resistance_runs = []

    for run_id in range(1, num_runs + 1):
        path_spy = f"{base_path}/run_{run_id}/coevolution_spy_ancestral_line_HoF.csv"
        path_res = f"{base_path}/run_{run_id}/coevolution_resistance_ancestral_line_HoF.csv"

        df_spy = pd.read_csv(path_spy)
        df_res = pd.read_csv(path_res)

        spy_tuples = [
            (int(row['generation']) + 1, [row[f'gene_{i}'] for i in range(10)])
            for _, row in df_spy.iterrows()
        ]
        res_tuples = [
            (int(row['generation']) + 1, [row[f'gene_{i}'] for i in range(14)])
            for _, row in df_res.iterrows()
        ]

        spy_runs.append(spy_tuples)
        resistance_runs.append(res_tuples)

    return spy_runs, resistance_runs

def run_full_ciao_all(spy_runs, resistance_runs, n_games=100):
    all_matrices = []
    n_generations = len(spy_runs[0])

    for run_id in range(1, 11):
        print(f"\n===== CIAO Random Players Run {run_id}/10 =====")
        spy_tuples = spy_runs[run_id - 1]
        res_tuples = resistance_runs[run_id - 1]

        ciao_matrix = np.zeros((n_generations, n_generations))

        for gx in range(n_generations):
            spy_gen, spy_genes = spy_tuples[gx]
            print(f"\n Spy Gen {spy_gen} ({gx+1}/{n_generations})")

            gen_start = time.time()

            for gy in range(n_generations):
                res_gen, res_genes = res_tuples[gy]

                if gy % 20 == 0:
                    print(f" → vs Resistance Gen {res_gen} ({gy+1}/{n_generations})")

                spy_wins = 0
                for _ in range(n_games):
                    # Build team: 1 resistance from gy + 2 co-players (resistance)
                    resistance_team = [res_genes] + random.choices(
                    [r[1] for r in res_tuples if r[0] != res_gen], k=2
                    )
                    spy_team = [spy_genes] + random.choices(
                    [s[1] for s in spy_tuples if s[0] != spy_gen], k=1
                    )
                    players = resistance_team + spy_team

                    winner = simulate_game(players)
                    if winner == 'spies':
                        spy_wins += 1

                ciao_matrix[gx, gy] = spy_wins / n_games

            print(f"✅ Spy Gen {spy_gen} done in {time.time() - gen_start:.2f}s")

        all_matrices.append(ciao_matrix)  

    return np.mean(all_matrices, axis=0)

# ======================= MAIN =======================
spy_runs, resistance_runs = load_ciao_data()
ciao_avg = run_full_ciao_all(spy_runs, resistance_runs)

# Save matrix
np.save("ciao_matrix_random_players.npy", ciao_avg)
print("✅ Saved final matrix as 'ciao_matrix_random_players.npy'")

# Plot
red_blue_cmap = LinearSegmentedColormap.from_list("RedBlue", ["blue", "red"])
plt.figure(figsize=(10, 8))
plt.imshow(
    ciao_avg,
    cmap=red_blue_cmap,
    origin='lower',
    extent=[1, 200, 1, 200],
    aspect='auto',
    vmin=0,
    vmax=1
)
plt.colorbar(label='Spy Win Rate')
plt.title("CIAO Matrix (10 Runs Averaged): Spy[gx] vs Resistance[gy]")
plt.xlabel("Spy Generation (gx)")
plt.ylabel("Resistance Generation (gy)")
plt.tight_layout()
plt.savefig("ciao_matrix_random_players_plot.png")
plt.show()
print(" Saved plot as 'ciao_matrix_random_players_plot.png'")
