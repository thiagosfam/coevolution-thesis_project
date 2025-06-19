import random
import numpy as np
from engine.game import simulate_game

# ─── Setup a small sample ────────────────────────────
# 3 resistance fighters (14-gene vectors), 3 spies (10-gene vectors)
rng = np.random.RandomState(42)
resistance_sample = rng.uniform(-1, 1, size=(3, 14)).tolist()
spy_sample = rng.random(size=(3, 10)).tolist()

# ─── Simulation function ─────────────────────────────
def run_batch(seed=None, n_games=50):
    """
    Play each res vs each spy for n_games, optionally seeding RNG once.
    Returns a list of (res_idx, spy_idx, wins) tuples.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    batch_results = []
    for i, r in enumerate(resistance_sample):
        for j, s in enumerate(spy_sample):
            wins = 0
            players = [r]*3 + [s]*2
            for _ in range(n_games):
                random.shuffle(players)
                if simulate_game(list_of_players=players, verbose=False) == "resistance":
                    wins += 1
            batch_results.append((i, j, wins))
    return batch_results

# ─── 1) With fixed seed ──────────────────────────────
first_seeded  = run_batch(seed=123)
second_seeded = run_batch(seed=123)
print("Seeded runs identical?", first_seeded == second_seeded)

# ─── 2) Without seeding at all ───────────────────────
first_unseeded  = run_batch(seed=None)
second_unseeded = run_batch(seed=None)
print("Unseeded runs identical?", first_unseeded == second_unseeded)

# Display a few results
print("\nA few results from seeded run:", first_seeded[:5])
print("\nA few results from seeded run:", first_seeded[:5])
