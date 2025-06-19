import random
import numpy as np
from typing import Any, Callable, List, Sequence
import pandas as pd
from engine.game import simulate_game

class LayeredParetoArchive:
    """
    A bounded, layered Pareto archive for coevolution.
    Keeps up to `max_layers` Pareto fronts of learners,
    and prunes testers that don't discriminate among those layers.
    """
    def __init__(self, max_layers: int):
        self.max_layers = max_layers
        self.learners: List[Any] = []
        self.testers:  List[Any] = []
        self.layers:   List[int] = []

    def update(
        self,
        learner_candidates: Sequence[Any],
        tester_candidates:  Sequence[Any],
        play_fn:  Callable[[Any, Any], str],
        role:     str
    ) -> None:
        """
        Merge in new learners & testers, peel Pareto layers for learners,
        keep layers 0…max_layers-1 in full, and prune testers.
        """
        # 1) Merge old + new
        all_learners = list(self.learners) + list(learner_candidates)
        all_testers  = list(self.testers)  + list(tester_candidates)

        # 2) Build payoff matrix F: learners × testers
        #    score=1 if learner wins, else 0
        F = np.zeros((len(all_learners), len(all_testers)))
        for i, L in enumerate(all_learners):
            for j, T in enumerate(all_testers):
                F[i, j] = 1 if play_fn(L, T) == role else 0

        # 3) Peel nondominated Pareto fronts among learners
        fronts = self._peel_fronts(F)

        # build a dict learner-idx -> layer-no
        layer_of = {i: ln for ln, front in enumerate(fronts) for i in front}

        # 4) Retain layers 0…max_layers-1 in full
        retained_idxs = []
        for layer_no, front in enumerate(fronts):
            if layer_no < self.max_layers:
                retained_idxs.extend(front)
            else:
                break

        # 5) Update learners
        self.learners = [all_learners[i] for i in retained_idxs]
        self.layers   = [layer_of[i]     for i in retained_idxs]

        # 6) Prune testers that don't discriminate
        self.testers = self._filter_testers(F, retained_idxs, all_testers)

    def _peel_fronts(self, F: np.ndarray) -> List[List[int]]:
        """
        Given payoff matrix F (learners x testers),
        return a list of Pareto-fronts (lists of learner-indices).
        """
        remaining = set(range(F.shape[0]))
        fronts: List[List[int]] = []

        def dominates(i: int, k: int) -> bool:
            return np.all(F[i] >= F[k]) and np.any(F[i] > F[k])

        while remaining:
            front = [
                i for i in remaining
                if not any(dominates(j, i) for j in remaining if j != i)
            ]
            fronts.append(front)
            remaining -= set(front)
        return fronts

    def _filter_testers(
        self,
        F:            np.ndarray,
        learner_idx:  List[int],
        all_testers:  List[Any]
    ) -> List[Any]:
        """
        Keep only those testers that assign different scores to at least
        one pair of learners in the same or adjacent retained layers.
        """
        # Recompute fronts & layer mapping among ALL learners
        fronts = self._peel_fronts(F)
        layer_of = {}
        for ln, front in enumerate(fronts):
            for i in front:
                layer_of[i] = ln

        # Build list of (idx, layer) only for retained learners
        retained = [(i, layer_of[i]) for i in learner_idx]

        kept = []
        for j, T in enumerate(all_testers):
            # Collect scores by layer
            scores_by_layer = {}
            for i, ln in retained:
                scores_by_layer.setdefault(ln, []).append(F[i, j])

            # Check discrimination in same layer
            discriminates = any(len(set(vals)) > 1
                                for vals in scores_by_layer.values())

            # Or between adjacent layers
            if not discriminates:
                layers = sorted(scores_by_layer)
                for a, b in zip(layers, layers[1:]):
                    if set(scores_by_layer[a]) != set(scores_by_layer[b]):
                        discriminates = True
                        break

            if discriminates:
                kept.append(T)

        return kept

    def sample_coplayers(self, k: int) -> List[Any]:
        """Uniformly sample k learners (with replacement)."""
        if not self.learners:
            return []
        return random.choices(self.learners, k=k)

    def sample_opponents(self, k: int) -> List[Any]:
        """Uniformly sample k tester (with replacement)."""
        return random.choices(self.testers, k=k)

    def export(self, filepath: str) -> None:
        """
        Export learner genomes to CSV, together with their Pareto-layer number
        (0 = first/front Pareto front, 1 = second front, …).
        """

        rows = []
        for idx, (L, layer) in enumerate(zip(self.learners, self.layers)):
            genes = getattr(L, 'genes', L)
            row = {'index': idx, 'layer': layer}
            for gi, g in enumerate(genes):
                row[f'gene_{gi}'] = g
            rows.append(row)

        pd.DataFrame(rows).to_csv(filepath, index=False)

def make_play_fn(role: str, n_games: int):
    def play_fn(learner: Any, tester: Any) -> str:
        spy_wins = 0
        for _ in range(n_games):
            if role == 'resistance':
                res_team = [learner] * 3
                spy_team = [tester] * 2
            else:
                spy_team = [learner] * 2
                res_team = [tester] * 3

            # DEBUG: Check gene lengths before simulation
            for p in res_team:
                assert len(p) == 14, f"Expected resistance gene length 14, got {len(p)}"
            for p in spy_team:
                assert len(p) == 10, f"Expected spy gene length 10, got {len(p)}"

            players = res_team + spy_team
            winner = simulate_game(players)
            if winner == 'spies':
                spy_wins += 1

        return 'spies' if spy_wins > n_games / 2 else 'resistance'

    return play_fn


# test_layered_pareto.py

def test_layered_pareto_archive():
    random.seed(0)

    # 1) Test learner pruning (max_layers=1)
    arch = LayeredParetoArchive(max_layers=1)
    learners = ['L1', 'L2']
    testers  = ['T1', 'T2']

    # L1 always wins, L2 always loses
    def play_fn(learner, tester):
        return 'spies' if learner == 'L1' else 'resistance'

    arch.update(learners, testers, play_fn, role='spies')
    # F = [[1,1],[0,0]] → front0=[L1], front1=[L2]
    assert arch.learners == ['L1'], f"Expected only ['L1'], got {arch.learners}"
    assert arch.testers == [],        f"Expected no testers, got {arch.testers}"
    print("✅ Learner pruning OK")

    # 2) Test tester discrimination (max_layers=2)
    arch = LayeredParetoArchive(max_layers=2)
    learners = ['L1', 'L2']
    testers  = ['T1', 'T2', 'T3']

    # T1/T2 discriminate (L1 wins, L2 loses), T3 does not (both win)
    def play_fn2(learner, tester):
        if tester == 'T3':
            return 'spies'
        return 'spies' if learner == 'L1' else 'resistance'

    arch.update(learners, testers, play_fn2, role='spies')
    # Both learners kept in front0
    assert set(arch.learners) == {'L1','L2'}, f"Expected both, got {arch.learners}"
    # Only T1/T2 discriminate → T3 dropped
    assert set(arch.testers) == {'T1','T2'}, f"Expected ['T1','T2'], got {arch.testers}"
    print("✅ Tester discrimination OK")

    print("\nAll LayeredParetoArchive smoke tests passed.")

if __name__ == '__main__':
    test_layered_pareto_archive()
