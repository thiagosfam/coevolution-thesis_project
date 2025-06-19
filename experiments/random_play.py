from engine.game_random import simulate_game


def run_baseline(n_games=10_000):
    win_rate = 0

    for game in range(n_games):
        winner = simulate_game()
        if winner == "resistance":
            win_rate += 1

    win_rate /= n_games
    print(f"Resistance win rate over {n_games} games: {win_rate * 100:.2f}%")


if __name__ == "__main__":
    run_baseline()
