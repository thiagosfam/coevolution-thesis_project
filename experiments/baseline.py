import random
import numpy as np
from typing import List, Tuple
random.seed(42)
np.random.seed(42)


class Player:
    def __init__(self, id: int, role: str):
        self.id = id
        self.role = role


def game_setup() -> Tuple[List[Player], Player, List[Player], List[Player]]:
    roles = ["resistance"] * 3 + ["spy"] * 2
    random.shuffle(roles)
    players = [Player(i, roles[i]) for i in range(5)]
    mission_leader = random.choice(players)
    return (players, mission_leader,
            [p for p in players if p.role == "spy"],
            [p for p in players if p.role == "resistance"])


def propose_team(players: List[Player], size: int) -> List[Player]:
    """Pure uniform random team selection"""
    return random.sample(players, size)


def execute_mission(team: List[Player]) -> bool:
    """Exact 50% sabotage probability per spy"""
    num_spies = sum(1 for p in team if p.role == "spy")
    sabotaged = random.random() < (1 - 0.5 ** num_spies)
    return not sabotaged


def simulate_game() -> str:
    players, mission_leader, spies, resistance = game_setup()
    mission_sizes = [2, 3, 2, 3, 3]  # Standard 5-player sequence
    successes = 0
    failures = 0

    for size in mission_sizes:
        # Simplified voting - assume team always approved
        team = propose_team(players, size)

        if execute_mission(team):
            successes += 1
        else:
            failures += 1

        if successes >= 3:
            return "resistance"
        if failures >= 3:
            return "spies"
    return "spies"  # Shouldn't reach here


def calculate_win_rate(n_games=100_000):
    wins = 0
    for _ in range(n_games):
        if simulate_game() == "resistance":
            wins += 1

    return wins / n_games


# Run analysis
print(f"Calculating win rate...")
win_rate = calculate_win_rate()
print(f"Resistance win rate: {win_rate:.3f}")