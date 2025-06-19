# tournament_full_archives.py
import numpy as np, pandas as pd, time
from pathlib import Path
from tqdm import tqdm
from evolutionary.pareto_archive import make_play_fn

# ─── Paths & filenames ────────────────────────────────────────────
CUSTOM_ROOT = Path("results/HoF_archive")
HOF_ROOT    = Path("results/HoF_archive")

RES_FILE = "resistance_archive.csv"      # in CustomDominanceArchive
SPY_FILE = "spy_archive.csv"             # in HoF_archive

RES_GENES, SPY_GENES = 14, 10
N_GAMES = 100
N_RUNS  = 10

def load_genomes(csv_path: Path, n_genes: int):
    df = pd.read_csv(csv_path)
    return df[[f"gene_{i}" for i in range(n_genes)]].to_numpy(float).tolist()

pop_scores = []

for run in range(1, N_RUNS + 1):
    res_csv = CUSTOM_ROOT / f"run_{run}" / RES_FILE
    spy_csv = HOF_ROOT    / f"run_{run}" / SPY_FILE

    if not res_csv.exists() or not spy_csv.exists():
        raise FileNotFoundError(f"Missing run {run} files.")

    R_pop = load_genomes(res_csv, RES_GENES)
    S_pop = load_genomes(spy_csv, SPY_GENES)

    print(f"\n── Run {run}:  {len(R_pop)} resistance × {len(S_pop)} spies ──")
    play = make_play_fn(role="resistance", n_games=N_GAMES)

    F = np.zeros((len(R_pop), len(S_pop)), dtype=np.uint8)

    start = time.time()
    for i, R in enumerate(tqdm(R_pop, desc=f"  rows {run}")):
        for j, S in enumerate(S_pop):
            F[i, j] = 1 if play(R, S) == "resistance" else 0
    elapsed = time.time() - start

    pop_win_rate = F.mean()
    pop_scores.append(pop_win_rate)
    print(f"  ► population win-rate = {pop_win_rate:.3%}   "
          f"({elapsed/60:.1f} min)")

    # save matrix
    out_dir = Path("results/master_tournaments"); out_dir.mkdir(exist_ok=True)
    stem = f"run{run:02d}_HoFRes_vs_HoFSpy"
    np.save(out_dir / f"{stem}.npy", F)
    pd.DataFrame(F).to_csv(out_dir / f"{stem}.csv", index=False)

# ─── summary ──────────────────────────────────────────────────────
mean = np.mean(pop_scores)
std  = np.std(pop_scores, ddof=1)
print("\n========  SUMMARY (10 runs)  ========")
print(f"Resistance (HoF) win-rate vs Spy (HoF): {mean:.2%}  ± {std:.2%} SD")
