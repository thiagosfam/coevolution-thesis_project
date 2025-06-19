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
        path_spy_ancestral_line = f"{base_path}/run_{run_id}/coevolution_spy_ancestral_line_HoF.csv"
        path_spy_HoF = f"{base_path}/run_{run_id}/coevolution_spy_hall_of_fame_HoF.csv"
        path_res_ancestral_line = f"{base_path}/run_{run_id}/coevolution_resistance_ancestral_line_HoF.csv"
        path_res_HoF = f"{base_path}/run_{run_id}/coevolution_resistance_hall_of_fame_HoF.csv"
       
        df_spy_ancestral_line = pd.read_csv(path_spy_ancestral_line)
        df_spy_HoF = pd.read_csv(path_spy_HoF)
        df_res_ancestral_line = pd.read_csv(path_res_ancestral_line)
        df_res_HoF = pd.read_csv(path_res_HoF)  
       
        spy_ancestral_line_tuples = [
            (int(row['generation']) + 1, [row[f'gene_{i}'] for i in range(10)])
            for _, row in df_spy_ancestral_line.iterrows()
        ]
        spy_HoF_tuples = [
            (None, [row[f'gene_{i}'] for i in range(10)])
            for _, row in df_spy_HoF.iterrows()
        ]
        res_ancestral_line_tuples = [
            (int(row['generation']) + 1, [row[f'gene_{i}'] for i in range(14)])
            for _, row in df_res_ancestral_line.iterrows()
        ]
        res_HoF_tuples = [
            (None, [row[f'gene_{i}'] for i in range(14)])
            for _, row in df_res_HoF.iterrows()
        ]

        spy_runs.append(spy_ancestral_line_tuples + spy_HoF_tuples)
        resistance_runs.append(res_ancestral_line_tuples + res_HoF_tuples)

    return spy_runs, resistance_runs

def run_master_tournament(spy_runs, resistance_runs, n_games=100):
    num_runs = len(spy_runs)
    num_generations = 200  # assume fixed for all runs

    spy_mt_all = []
    res_mt_all = []

    for run_id in range(num_runs):
        print(f"\n===== Master Tournament Run {run_id + 1}/{num_runs} =====")

        spy_line = spy_runs[run_id][:num_generations]
        spy_HoF = [ind for _, ind in spy_runs[run_id][num_generations:]]

        res_line = resistance_runs[run_id][:num_generations]
        res_HoF = [ind for _, ind in resistance_runs[run_id][num_generations:]]

        spy_mt_scores = []
        res_mt_scores = []

        for g in range(num_generations):
            # === Spy[g] vs ALL resistance HoF ===
            spy_gen, spy_genes = spy_line[g]
            spy_wins = 0
            total_games = 0

            for res_genes in res_HoF:
                players = [res_genes] * 3 + [spy_genes] * 2
                for _ in range(n_games):
                    random.shuffle(players)
                    if simulate_game(players) == 'spies':
                        spy_wins += 1
                    total_games += 1

            spy_mt_scores.append(spy_wins / total_games if total_games else 0)

            # === Resistance[g] vs ALL spy HoF ===
            res_gen, res_genes = res_line[g]
            res_wins = 0
            total_games = 0

            for spy_genes in spy_HoF:
                players = [res_genes] * 3 + [spy_genes] * 2
                for _ in range(n_games):
                    random.shuffle(players)
                    if simulate_game(players) == 'resistance':
                        res_wins += 1
                    total_games += 1

            res_mt_scores.append(res_wins / total_games if total_games else 0)

            if g % 20 == 0:
                print(f"  Generation {g+1}/{num_generations} - Spy MT: {spy_mt_scores[-1]:.2f} | Resistance MT: {res_mt_scores[-1]:.2f}")

        spy_mt_all.append(spy_mt_scores)
        res_mt_all.append(res_mt_scores)

    # Convert to np arrays and average over runs
    spy_mt_avg = np.mean(spy_mt_all, axis=0)
    res_mt_avg = np.mean(res_mt_all, axis=0)

    return spy_mt_avg, res_mt_avg

# Run it
spy_runs, resistance_runs = load_ciao_data()
spy_mt, res_mt = run_master_tournament(spy_runs, resistance_runs)

# Plot
gens = np.arange(1, 201)
plt.figure(figsize=(10, 6))
plt.plot(gens, spy_mt, label='Spy', color='red')
plt.plot(gens, res_mt, label='Resistance Figther', color='blue')


plt.xlabel("Generation")
plt.ylabel("Average Win Rate vs HoF")
plt.title("Master Tournament Evaluation (Averaged over 10 Runs)")
plt.legend()
plt.tight_layout()
plt.savefig("master_tournament_plot.png")
plt.show()
print("✅ Saved plot as 'master_tournament_plot.png'")
