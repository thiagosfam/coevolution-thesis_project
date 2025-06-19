from abc import ABC, abstractmethod
import random
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Set
from engine.game import simulate_game
from scipy.spatial.distance import hamming
import json

class Archive(ABC):
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size

    @abstractmethod
    def update(self, individual, fitness, generation):
        """Add the generation's champion to the archive."""
        pass

    @abstractmethod
    def sample_coplayers(self, role, k):
        """Return k same-role players from the archive."""
        pass

    @abstractmethod
    def sample_opponents(self, role, k):
        """Return k opposite-role players from the archive."""
        pass

    @abstractmethod
    def export(self, filepath: str):
        """
        Write the entire archive contents (and any meta-info like generation)
        to a CSV at `filepath`.
        """
        pass

class HallOfFameArchive(Archive):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.archive = []  # list of tuples (genes, fitness, generation)

    def update(self, individual, fitness, generation):
        # Add the new entry
        self.archive.append((individual, fitness, generation))
        # Sort by fitness descending
        self.archive.sort(key=lambda entry: entry[1], reverse=True)
        # Keep only the top max_size
        if len(self.archive) > self.max_size:
            # Drop the worst
            self.archive = self.archive[:self.max_size]

    def sample_coplayers(self, k):
        pool = [ind for ind, _, _ in self.archive]
        return random.choices(pool, k=k)

    def sample_opponents(self, k):
        return self.sample_coplayers(k)
    
    def get_archive(self):
        pool = [ind for ind, _, _ in self.archive]
        return pool

    def export(self, filepath: str):
        import pandas as pd
        rows = []
        for indiv, fit, gen in self.archive:
            row = {'generation': gen, 'fitness': fit}
            for i, g in enumerate(indiv):
                row[f'gene_{i}'] = g
            rows.append(row)
        pd.DataFrame(rows).to_csv(filepath, index=False)


class BestOfGen3HoF(Archive):
    """
    Keeps the top 3 individuals (by fitness) from each generation.
    When sampling, draws uniformly from all archived elites.
    """
    def __init__(self):
        super().__init__(max_size=None)
        self.by_gen = {} 

    def update(self, individual, fitness, generation):
        # Ensure a list exists for this generation
        self.by_gen.setdefault(generation, []).append((individual, fitness))
        # Keep only the top-3 by descending fitness
        self.by_gen[generation] = sorted(
            self.by_gen[generation],
            key=lambda x: x[1],
            reverse=True
        )[:3]

    def _all(self):
        # Flatten all per-gen lists into one list of individuals
        return [ind for gen_list in self.by_gen.values() for ind, _ in gen_list]

    def sample_coplayers(self, k):
        pool = self._all()
        # Sample k coplayers with replacement
        return random.choices(pool, k=k)

    def sample_opponents(self, k):
        # identical behavior in this simple policy
        return self.sample_coplayers(k)
    
    def export(self, filepath: str):
        # Flatten into rows: generation, fitness, gene_0..gene_n
        rows = []
        for gen, entries in sorted(self.by_gen.items()):
            for indiv, fit in entries:
                row = {'generation': gen, 'fitness': fit}
                # assume indiv is iterable of gene values
                for i, g in enumerate(indiv):
                    row[f'gene_{i}'] = g
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

class MaxMinDiverseArchive:
    """
    Maintains up to max_size diverse individuals with highest fitness.
    When full, replaces the individual whose removal + insertion of the new
    candidate yields the greatest improvement in minimum pairwise distance.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.archive = []  # List of tuples: (genes: np.ndarray, fitness: float, generation: int)

    def _min_pairwise_distance(self, gene_list):
        dists = [
            np.linalg.norm(g1 - g2)
            for i, g1 in enumerate(gene_list)
            for j, g2 in enumerate(gene_list)
            if i < j
        ]
        return min(dists) if dists else 0.0

    def update(self, individual, fitness, generation):
        genes = np.array(individual, copy=False, dtype=float)

        if len(self.archive) < self.max_size:
            self.archive.append((genes, fitness, generation))
            return

        # If not better than the worst, skip
        worst_fitness = min(self.archive, key=lambda x: x[1])[1]
        if fitness <= worst_fitness:
            return

        # Evaluate replacement of each current individual
        best_min_dist = -np.inf
        best_index = None

        for i in range(self.max_size):
            candidate_genes = [g for j, (g, _, _) in enumerate(self.archive) if j != i]
            candidate_genes.append(genes)
            min_dist = self._min_pairwise_distance(candidate_genes)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_index = i

        if best_index is not None:
            self.archive[best_index] = (genes, fitness, generation)

    def sample_coplayers(self):
        pool = [genes for (genes, _fit, _gen,) in  self.archive]
        return random.choice(pool)

    def sample_opponents(self): 
        return self.sample_coplayers()

    def get_archive(self):
        pool = [genes for (genes, _fit, _gen,) in  self.archive]
        return pool

    def export(self, filepath: str):
        import pandas as pd
        rows = []
        for g, f, gen in self.archive:
            row = {'generation': gen, 'fitness': f}
            for i, g_val in enumerate(g):
                row[f'gene_{i}'] = g_val
            rows.append(row)
        pd.DataFrame(rows).to_csv(filepath, index=False)

    
class QualityDiversityArchive(Archive):
    """
    A bounded quality-diversity archive:
      - Keeps up to `max_size` entries.
      - Each entry has (genes, fitness, novelty, generation, layer).

    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        # entries: (genes: np.array, fitness: float, novelty: float, generation: int, layer: int)
        self.archive = []
    
    def update(self, individual, fitness, generation=None):
        # 0) Shortcut to local variables
        archive = self.archive
        max_size = self.max_size

        # 1) Compute novelty for P vs all existing archive entries
        P = np.array(individual, dtype=float)
        dists = [np.linalg.norm(P - entry[0]) for entry in archive]  # O(N)

        novelty_P = float(min(dists)) if dists else float('inf')

        # 2) Update old entries' novelty in place
        for idx, dist in enumerate(dists):
            g, f, old_nov, gen, _ = archive[idx]
            # take the smaller of (old novelty) and (distance to P)
            archive[idx] = (g, f, min(old_nov, dist), gen, None)

        # 3) Append P with its novelty (layer placeholder = None)
        archive.append((P, fitness, novelty_P, generation, None))

        # 4) Build objective matrix for Pareto (fitness ↑, novelty ↑)
        F = np.array([[f, nov] for (_, f, nov, _, _) in archive])

        # 5) Peel Pareto fronts
        remain = set(range(len(archive)))
        fronts = []
        def dominates(i, j):
            return np.all(F[i] >= F[j]) and np.any(F[i] > F[j])

        while remain:
            front = [i for i in remain
                 if not any(dominates(j, i) for j in remain if j != i)]
            fronts.append(front)
            remain -= set(front)

        # 6) Record each index's layer
        layer_of = {}
        for layer_idx, front in enumerate(fronts):
            for idx in front:
                layer_of[idx] = layer_idx

        # 7) Truncate to max_size, whole-front-first
        kept = []
        count = 0
        for front in fronts:
            if count + len(front) <= max_size:
                kept.extend(front)
                count += len(front)
            else:
                slots = max_size - count
                kept.extend(random.sample(front, slots))
                break

        # 8) Rebuild archive with proper layer values
        new_archive = []
        for i in kept:
            g, f, nov, gen, _ = archive[i]
            lay = layer_of[i]
            new_archive.append((g, f, nov, gen, lay))

        self.archive = new_archive

    def sample_coplayers(self, k):
        pool = [genes for (genes, _fit, _nov, _gen, _lay) in  self.archive]
        return random.choices(pool, k=k)

    def sample_opponents(self, k): 
        return self.sample_coplayers(k)

    def export(self, filepath: str):
        import pandas as pd
        rows = []
        for genes, fit, nov, gen, lay in self.archive:
            row = {
                'generation': gen,
                'layer':      lay,
                'fitness':    fit,
                'novelty':    nov
            }
            for i, g in enumerate(genes):
                row[f'gene_{i}'] = g
            rows.append(row)
        pd.DataFrame(rows).to_csv(filepath, index=False)

class _CustomArchive(ABC):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.resistance_archive = []  # list of tuples (genes, fitness, generation) for resistance fighters
        self.spies_archive = [] # list of tuples (genes, fitness, generation) for spies
        half = max_size // 2
        self.fitness_matrix = np.zeros((half, half))
        self.win_matrix = np.zeros((half, half))
        self.threshold = 0.6

    def update(self, individual_r, fitness_r, individual_s, fitness_s, generation):

        if len(self.resistance_archive) < self.max_size // 2 and len(self.spies_archive) < self.max_size // 2:
            self.resistance_archive.append((individual_r, fitness_r, generation))
            self.spies_archive.append((individual_s, fitness_s, generation))
            return
        
    
        print(f"[DEBUG] Resistance size: {len(self.resistance_archive)}, Spies size: {len(self.spies_archive)}")
        print("[DEBUG] Checking if fitness_matrix still has zeros...")
        if np.any(self.fitness_matrix == 0):
            resistance = [genes for genes, _fitness, _generation in self.resistance_archive]
            spies = [genes for genes, _fitness, _generation in self.spies_archive]

            for i, R in enumerate(resistance):
                for j, S in enumerate(spies):
                    result = self.play_fn(L=R, T=S, role='resistance', n_games=100)
                    self.fitness_matrix[i, j] = result
                    self.win_matrix[i,j] = 1 if result > 0.6 else 0

            print("[DEBUG] Matrix populated.")
            print("[DEBUG] Any zeros in active fitness_matrix?:", np.any(self.fitness_matrix[:len(self.resistance_archive), :len(self.spies_archive)] == 0))
            print("[DEBUG] Fitness Matrix shape.", self.fitness_matrix.shape)
            print("[DEBUG] Win Matrix shape.", self.win_matrix.shape)

            # Test new resistance candidate against the entire archive
            r_results = [
                self.play_fn(L=individual_r, T=spy[0], role='resistance', n_games=100)
                for spy in self.spies_archive
            ]
            res_candidate_fitness = np.mean(r_results)
            new_win_vector_r = [1 if score > 0.6 else 0 for score in r_results]

            # Test new spy candidate against the entire archive
            s_results = [
                self.play_fn(L=individual_s, T=resistance[0], role='spy', n_games=100)
                for resistance in self.resistance_archive
            ]
            spy_candidate_fitness = np.mean(s_results)
            new_win_vector_s = [1 if (1.0 - score) > 0.6 else 0 for score in s_results]

            res_fitness = np.mean(self.fitness_matrix, axis=1)
            min_res_fitness = np.min(res_fitness)
            min_res_index = np.argmin(res_fitness)

            spy_fitness = np.mean(1.0 - self.fitness_matrix, axis=0)
            min_spy_fitness = np.min(spy_fitness)
            min_spy_index = np.argmin(spy_fitness)

            print(f"[Resistance] Candidate fitness: {res_candidate_fitness:.3f}, Worst in archive: {min_res_fitness:.3f}")
            print(f"[Spy]        Candidate fitness: {spy_candidate_fitness:.3f}, Worst in archive: {min_spy_fitness:.3f}")
            print(f"[Resistance] Win vector: {new_win_vector_r}")
            print(f"[Spy]        Win vector: {new_win_vector_s}")

            if res_candidate_fitness > min_res_fitness and self.is_diverse(new_win_vector_r, role='resistance'):
                print("[DEBUG] Candidate passed fitness check — checking diversity.")
                self.resistance_archive[min_res_index] = (individual_r, fitness_r, generation)
                for j, spy in enumerate(self.spies_archive):
                    result = r_results[j]
                    self.fitness_matrix[min_res_index, j] = result
                    self.win_matrix[min_res_index, j] = 1 if result > 0.6 else 0

            if spy_candidate_fitness > min_spy_fitness and self.is_diverse(new_win_vector_s, role='spy'):
                self.spies_archive[min_spy_index] = (individual_s, fitness_s, generation)
            

    def is_diverse(self, win_vector, role):

        if role == 'resistance':
            archive_vectors = self.win_matrix[:len(self.resistance_archive), :len(self.spies_archive)]
        elif role == 'spy':
            archive_vectors = 1 - self.win_matrix[:len(self.resistance_archive), :len(self.spies_archive)]
            archive_vectors = archive_vectors.T
        else:
            raise ValueError("Role must be 'resistance' or 'spy'")

        # Compare to all archive vectors
        for idx, archived in enumerate(archive_vectors):
            dist = hamming(win_vector, archived)
            print(f"Comparing to archive[{idx}] - Hamming: {dist:.3f}")
            if dist < 0.1:
                print("❌ Not diverse: match found.")
                return False
        print("✅ Diverse: no close match found.")
        return True

    def play_fn(self, L, T, role, n_games):
        win = 0
        
        if role == 'resistance':
            learners = [L] * 3
            testers = [T] * 2
        else:
            learners = [L] * 2
            testers = [T] * 3

        players = learners + testers
        
        for _ in range(n_games):
            winner = simulate_game(players, verbose=False)
            if winner == 'resistance':
                win += 1

        return win / n_games

class _CustomArchivePerformance(ABC):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.resistance_archive = []  # list of tuples (genes, fitness, generation) for resistance fighters
        self.spies_archive = [] # list of tuples (genes, fitness, generation) for spies
        half = max_size // 2
        self.resistance_fitness_matrix = np.zeros((half, half))
        self.resistance_win_matrix = np.zeros((half, half))
        self.spy_fitness_matrix = np.zeros((half, half))
        self.spy_win_matrix = np.zeros((half, half))
        self.threshold = 0.6
        self.spy_replacement_counter = 0
        self.resistance_replacement_counter = 0
        self.matrix_built = False

    def dominates(self, candidate_vector, archived_vector):
        return np.all(np.array(candidate_vector) > np.array(archived_vector))

    def update(self, individual_r, fitness_r, individual_s, fitness_s, generation):

        if len(self.resistance_archive) < self.max_size // 2 and len(self.spies_archive) < self.max_size // 2:
            self.resistance_archive.append((individual_r, fitness_r, generation))
            self.spies_archive.append((individual_s, fitness_s, generation))
            return
        
        #print(f"[DEBUG] Resistance size: {len(self.resistance_archive)}, Spies size: {len(self.spies_archive)}")
        #print("[DEBUG] Checking if fitness_matrix still has zeros...")

        r_size = len(self.resistance_archive)
        s_size = len(self.spies_archive)
        
        if not self.matrix_built:
            resistance = [genes for genes, _fitness, _generation in self.resistance_archive]
            spies = [genes for genes, _fitness, _generation in self.spies_archive]

            for i, R in enumerate(resistance):
                for j, S in enumerate(spies):
                    result = self.play_fn(L=R, T=S, role='resistance', n_games=100)
                    self.resistance_fitness_matrix[i,j] = result
                    self.resistance_win_matrix[i,j] = 1 if result > self.threshold else 0
                    self.spy_fitness_matrix[j,i] = 1 - result
                    self.spy_win_matrix[j,i] = 1 if result < (1 - self.threshold) else 0

            self.matrix_built = True

            # Update fitness of resistance and spies

            for i in range(r_size):
                old_fit = self.resistance_archive[i][1]
                new_fit = np.mean(self.resistance_fitness_matrix[i, :s_size])
                #print(f"[DEBUG] Resistance {i}: Fitness updated {old_fit:.3f} → {new_fit:.3f}")
                self.resistance_archive[i] = (self.resistance_archive[i][0], new_fit, self.resistance_archive[i][2])


            for j in range(s_size):
                old_fit = self.spies_archive[j][1]
                new_fit = np.mean(self.spy_fitness_matrix[j, :r_size])
                #print(f"[DEBUG] Spy {j}: Fitness updated {old_fit:.3f} → {new_fit:.3f}")
                self.spies_archive[j] = (self.spies_archive[j][0], new_fit, self.spies_archive[j][2])

            """print("[DEBUG] Matrix populated.")
            print("[DEBUG] Any zeros in active fitness_matrix?:", np.any(self.fitness_matrix[:len(self.resistance_archive), :len(self.spies_archive)] == 0))
            print("[DEBUG] Fitness Matrix shape.", self.fitness_matrix.shape)
            print("[DEBUG] Win Matrix shape.", self.win_matrix.shape)"""

        # Test new resistance candidate against the entire archive
        r_results = [
            self.play_fn(L=individual_r, T=spy[0], role='resistance', n_games=100)
            for spy in self.spies_archive
        ]
        res_candidate_fitness = np.mean(r_results)
        new_win_vector_r = [1 if score > self.threshold else 0 for score in r_results]
        num_win_new_res = sum(new_win_vector_r)

        # Test new spy candidate against the entire archive
        s_results = [
            self.play_fn(L=individual_s, T=resistance[0], role='spy', n_games=100)
            for resistance in self.resistance_archive
        ]

        spy_candidate_fitness = np.mean(s_results)
        new_win_vector_s = [1 if (1.0 - score) > self.threshold else 0 for score in s_results]
        num_win_new_spy = sum(new_win_vector_s)
        
        # Get the index of resistance fighter with the least wins
        row_wins_res = self.resistance_win_matrix[:r_size, :s_size].sum(axis=1)
        min_res_idx  = int(np.argmin(row_wins_res))
        num_win_old_res    = int(row_wins_res[min_res_idx])
        min_res_fitness    = float(self.resistance_fitness_matrix[min_res_idx, :s_size].mean())


        # Get the index of spy with the least wins
        row_wins_spy = self.spy_win_matrix[:s_size, :r_size].sum(axis=1)
        min_spy_idx  = int(np.argmin(row_wins_spy))
        num_win_old_spy    = int(row_wins_spy[min_spy_idx])
        min_spy_fitness    = float(self.spy_fitness_matrix[min_spy_idx, :r_size].mean())

        """print(f"[Resistance] Candidate fitness: {res_candidate_fitness:.3f}, Worst in archive: {min_res_fitness:.3f}")
        print(f"[Spy]        Candidate fitness: {spy_candidate_fitness:.3f}, Worst in archive: {min_spy_fitness:.3f}")
        print(f"[Resistance] Candidate win vector: {new_win_vector_r}")
        print(f"[Resistance] Worst in archive win vector: {min_res_win_vector}")
        print(f"[Spy]        Candidate win vector: {new_win_vector_s}")
        print(f"[Spy]        Worst in archive win vector: {min_spy_win_vector}")"""

        #  (num_win_new_res > num_win_old_res) or (num_win_new_res == num_win_old_res and res_candidate_fitness > min_res_fitness)

        if num_win_new_res > num_win_old_res or (num_win_new_res == num_win_old_res and res_candidate_fitness > min_res_fitness):
                print("✅ Resistance candidate is better than min archived resistance")
                self.resistance_archive[min_res_idx] = (individual_r, res_candidate_fitness, generation)
                self.resistance_replacement_counter += 1

                # Update oppoenents vectors to reflect performance against new  archive entrant
                for j, _ in enumerate(self.spies_archive):
                    result = r_results[j]
                    self.resistance_fitness_matrix[min_res_idx, j] = result
                    self.resistance_win_matrix[min_res_idx, j] = 1 if result > self.threshold else 0
                    self.spy_fitness_matrix[j, min_res_idx] = 1 - result
                    self.spy_win_matrix[j, min_res_idx]     = int((1 - result) > self.threshold)

                    
        # (num_win_new_spy > num_win_old_spy) or (num_win_new_spy == num_win_old_spy and spy_candidate_fitness > min_spy_fitness)

        if num_win_new_spy > num_win_old_spy or (num_win_new_spy == num_win_old_spy and spy_candidate_fitness > min_spy_fitness):
                print("✅ Spy candidate is better than min archived spy")
                self.spies_archive[min_spy_idx] = (individual_s, spy_candidate_fitness, generation)
                self.spy_replacement_counter += 1

                for i, _ in enumerate(self.resistance_archive):
                    result = s_results[i]
                    spy_win = int((1 - result) > self.threshold)   # 1 if spy wins, else 0
                    self.spy_fitness_matrix[min_spy_idx, i] = 1 - result
                    self.spy_win_matrix[min_spy_idx, i] = spy_win
                    # mirror bit for the resistance matrices
                    self.resistance_fitness_matrix[i, min_spy_idx] = result
                    self.resistance_win_matrix[i, min_spy_idx] = 1 - spy_win

        assert np.allclose(self.resistance_fitness_matrix[:r_size,:s_size] +
                   self.spy_fitness_matrix[:s_size,:r_size].T, 1.0)

            
    def play_fn(self, L, T, role, n_games):
        win = 0
        
        if role == 'resistance':
            learners = [L] * 3
            testers = [T] * 2
        else:
            learners = [L] * 2
            testers = [T] * 3

        players = learners + testers
        
        for _ in range(n_games):
            winner = simulate_game(players, verbose=False)
            if winner == 'resistance':
                win += 1

        return win / n_games
    
    def get_resistance_archive(self):
        return [genes for genes, *_ in self.resistance_archive]

    def get_spies_archive(self):
        return [genes for genes, *_ in self.spies_archive]

    def sample_resistance(self):
        return random.choice(self.get_resistance_archive())

    def sample_spy(self):
        return random.choice(self.get_spies_archive())
    
    def export_archives(self, res_path: str, spy_path: str):
        res_rows = []
        for genes, fitness, generation in self.resistance_archive:
            row = {
                'generation': generation,
                'fitness': fitness,
            }
            for i, g in enumerate(genes):
                row[f'gene_{i}'] = g
            res_rows.append(row)
        pd.DataFrame(res_rows).to_csv(res_path, index=False)

        spy_rows = []
        for genes, fitness, generation in self.spies_archive:
            row = {
                'generation': generation,
                'fitness': fitness,
            }
            for i, g in enumerate(genes):
                row[f'gene_{i}'] = g
            spy_rows.append(row)
        pd.DataFrame(spy_rows).to_csv(spy_path, index=False)

    def export_matrices(self, folder: str) -> None:
        """
        Save the current learner-tester matrices to four .npy files
        inside *folder*:

            res_fitness.npy   resistance_fitness_matrix (r x s, float)
            spy_fitness.npy   spy_fitness_matrix        (s x r, float)
            res_win.npy       resistance_win_matrix     (r x s, uint8)
            spy_win.npy       spy_win_matrix            (s x r, uint8)
        """
        os.makedirs(folder, exist_ok=True)

        r = len(self.resistance_archive)
        s = len(self.spies_archive)

        np.save(os.path.join(folder, "res_fitness.npy"),
                self.resistance_fitness_matrix[:r, :s])
        np.save(os.path.join(folder, "spy_fitness.npy"),
                self.spy_fitness_matrix[:s, :r])
        np.save(os.path.join(folder, "res_win.npy"),
                self.resistance_win_matrix[:r, :s].astype(np.uint8))
        np.save(os.path.join(folder, "spy_win.npy"),
                self.spy_win_matrix[:s, :r].astype(np.uint8))

        print(f"[Archive] matrices exported to {os.path.abspath(folder)}")

    def print_best_spy(self):
        """
        Report the spy in the archive that wins against the largest number
        of resistance fighters.

        Assumes:
            • spy_win_matrix rows  = spies   (len = s_size)
            • spy_win_matrix cols  = resistance fighters (len = r_size)
            • entries are 0/1  (1 = spy wins that matchup)
        """
        s_size = len(self.spies_archive)          # active spies
        r_size = len(self.resistance_archive)     # active resistance

        if s_size == 0 or r_size == 0:
            print("Archive is empty.")
            return

        # 1) wins per spy  (row-wise sum)
        wins_per_spy = self.spy_win_matrix[:s_size, :r_size].sum(axis=1)

        # 2) index of the spy with the most wins
        best_spy_idx = int(np.argmax(wins_per_spy))      # 0 … s_size-1

        # 3) retrieve data
        best_spy_vector = self.spy_win_matrix[best_spy_idx, :r_size]
        best_spy_genome = self.spies_archive[best_spy_idx]
        num_wins        = int(wins_per_spy[best_spy_idx])
        win_rate        = num_wins / r_size

        # 4) print nicely
        print(f"─ Best spy (row {best_spy_idx}) ─────────────────────────")
        print(f"Genome:  {best_spy_genome}")
        print(f"Wins vs resistance: {num_wins}/{r_size}  ({win_rate:.2%})")

    def print_best_resistance(self):
        """
        Report the resistance fighter that defeats the largest number of
        archived spies.

        Assumes:
            • resistance_win_matrix rows = resistance fighters (len = r_size)
            • resistance_win_matrix cols = spies              (len = s_size)
            • entries are 0/1  (1 = resistance wins that matchup)
        """
        r_size = len(self.resistance_archive)     # active resistance fighters
        s_size = len(self.spies_archive)          # active spies

        if r_size == 0 or s_size == 0:
            print("Archive is empty.")
            return

        # 1) wins per resistance fighter  (row-wise sum)
        wins_per_res = self.resistance_win_matrix[:r_size, :s_size].sum(axis=1)

        # 2) index of the fighter with the most wins
        best_res_idx = int(np.argmax(wins_per_res))          # 0 … r_size-1

        # 3) retrieve data
        best_res_vector = self.resistance_win_matrix[best_res_idx, :s_size]
        best_res_genome = self.resistance_archive[best_res_idx]
        num_wins        = int(wins_per_res[best_res_idx])
        win_rate        = num_wins / s_size

        # 4) print nicely
        print(f"─ Best resistance fighter (row {best_res_idx}) ───────────")
        print(f"Genome:  {best_res_genome}")
        print(f"Wins vs spies: {num_wins}/{s_size}  ({win_rate:.2%})")


class _Individual():
    def __init__(self, genes, i_fitness, generation):
        self.genes = genes
        self.generation = generation
        self.i_fitness = i_fitness  # insertion fitness
        self.a_fitness = 0  # archive fitness
        self.id = generation
        self.won_ids = []
        self.lost_ids = []
        self.played_against = []
        self.wins = 0
    
class CustomArchivePerformance:
    def __init__(self, role: str, max_size: int, threshold: float = 0.6):
        self.role       = role            # 'resistance' or 'spies'
        self.max_size   = max_size
        self.threshold  = threshold
        self.members: list[_Individual] = []

    # 1) seed with an initial batch (at generation 0 only)
    def seed(self, batch: list[_Individual], opponent_batch: list[_Individual]):
        for ind in batch:
            self._evaluate_vs_batch(ind, opponent_batch)
            self._insert_without_check(ind)
        print(f"[seed] {self.role}: {len(self.members)} individuals")

    # 2) add a *single* candidate each generation
    def add_or_replace(self, cand: _Individual, opponent_archive: "Archive"):
        
        # evaluate cand vs ALL opponents currently in the other archive
        self._evaluate_vs_batch(cand, opponent_archive.members)

        if len(self.members) < self.max_size:            # room left
            self._insert_without_check(cand)
            return

        # identify worst current member
        worst_idx = min(
            range(self.max_size),
            key=lambda i: (self.members[i].wins,
                           self.members[i].a_fitness)  )  # tie-break

        if (cand.wins > self.members[worst_idx].wins or
            (cand.wins == self.members[worst_idx].wins and
             cand.a_fitness > self.members[worst_idx].a_fitness)):
            self.members[worst_idx] = cand   # overwrite, index stays valid

    # ---------- internals ----------
    def _insert_without_check(self, ind: _Individual):
        if len(self.members) >= self.max_size:
            raise RuntimeError("archive full; call add_or_replace()")
        self.members.append(ind)

    def _evaluate_vs_batch(self, ind: _Individual, opponents: list[_Individual]):
        if not opponents:          # first few seeds
            ind.wins = ind.played_against = 0
            ind.a_fitness = 0.0
            return

        wins = 0
        total_rate = 0.0
        for opp in opponents:
            #print(f"ind genes: {ind.genes}")
            #print(f"opp genes: {opp.genes}")
            rate = self.play_fn(L=ind.genes, T=opp.genes, n_games=100)
            total_rate += rate

            if rate > self.threshold:
                wins += 1
                ind.won_ids.append(opp.generation)    # or opp.id

            else:
                ind.lost_ids.append(opp.generation)

        ind.wins       = wins
        ind.played_against  = len(opponents)
        ind.a_fitness  = total_rate / ind.played_against
    

    def play_fn(self, L, T, n_games):
        win = 0

        if self.role == 'resistance':
            learners = [L] * 3
            testers = [T] * 2
        elif self.role == 'spies':
            learners = [L] * 2
            testers = [T] * 3

        players = learners + testers
        
        for _ in range(n_games):
            winner = simulate_game(players, verbose=False)
            if winner == self.role:
                win += 1
            

        return win / n_games

    def get_archive(self):
        """
        Return a **list** of genome arrays currently stored in the archive
       
        """
        return [ind.genes for ind in self.members]
    
    def export(self, csv_path: str):
        """
        Write the current archive to `csv_path`.

        Each row = one Individual with all its statistics and genes.
        """
        rows = []
        for ind in self.members:
            row = {
                "generation": ind.generation,
                "wins"      : ind.wins,
                "opponents" : ind.played_against,
                "a_fitness" : ind.a_fitness,
                "i_fitness" : ind.i_fitness,          # optional but nice
                "won_ids"   : json.dumps(ind.won_ids),
                "lost_ids"  : json.dumps(ind.lost_ids)
            }
            for i, g in enumerate(ind.genes):
                row[f"gene_{i}"] = g
            rows.append(row)

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[export] wrote {len(rows)} rows → {csv_path}")

class CustomArchiveExpand:
    def __init__(self, role: str, max_size: int, threshold: float = 0.6):
        self.role       = role            # 'resistance' or 'spies'
        self.max_size   = max_size
        self.threshold  = threshold
        self.members: list[_Individual] = []

    # 1) seed with an initial batch (at generation 0 only)
    def seed(self, batch: list[_Individual], opponent_batch: list[_Individual]):
        for ind in batch:
            self._evaluate_vs_batch(ind, opponent_batch)
        print(f"[seed] {self.role}: {len(self.members)} individuals")

    # 2) add a *single* candidate each generation
    def add_or_replace(self, candidate: _Individual, opponent_archive: Archive):
    # 1) Evaluate candidate against every opponent in the OTHER archive
        self._evaluate_vs_batch(candidate, opponent_archive.members)
        cand_wins = set(candidate.won_ids)  # e.g. {0,3,5,8,...}

    # 2) Build the union of all wins in your existing archive
        all_archive_wins = set().union(*(ind.won_ids for ind in self.members))

    # 3a) Case A: candidate conquers *new* ground
        if not cand_wins.issubset(all_archive_wins):
            accept = True

        else:
            # 3b) Case B: candidate wins exactly the same set,
            #        but is measurably stronger on at least one common opponent
            accept = False
            for opp_id in cand_wins:
                # find the archive member who currently has the best win‐rate vs opp_id
                best_old_rate = max(
                    play_rate
                    for ind in self.members
                    for oid, play_rate in ind.win_rates.items()
                    if oid == opp_id
                )
                # compare candidate’s rate
                cand_rate = candidate.win_rates[opp_id]
                if cand_rate > best_old_rate:
                    accept = True
                    break

        if not accept:
            return   # reject the candidate outright

        # 4) Otherwise, it passed our “adds new challenge” test.
        #    Now evict your usual “worst” (or apply your other replacement rule)
        worst_idx = min(
            range(len(self.members)),
            key=lambda i: (self.members[i].wins, self.members[i].a_fitness)
        )
        self.members[worst_idx] = candidate
    # ---------- internals ----------
    def _insert_without_check(self, ind: _Individual):
        if len(self.members) >= self.max_size:
            raise RuntimeError("archive full; call add_or_replace()")
        self.members.append(ind)

    def _evaluate_vs_batch(self, ind: _Individual, opponents: list[_Individual]):
        if not opponents:          # first few seeds
            ind.wins = ind.played_against = 0
            ind.a_fitness = 0.0
            return

        wins = 0
        total_rate = 0.0
        for opp in opponents:
            #print(f"ind genes: {ind.genes}")
            #print(f"opp genes: {opp.genes}")
            rate = self.play_fn(L=ind.genes, T=opp.genes, n_games=100)
            total_rate += rate

            if rate > self.threshold:
                wins += 1
                ind.won_ids.append(opp.generation)    # or opp.id

            else:
                ind.lost_ids.append(opp.generation)

        ind.wins       = wins
        ind.played_against  = len(opponents)
        ind.a_fitness  = total_rate / ind.opponents
    

    def play_fn(self, L, T, n_games):
        win = 0

        if self.role == 'resistance':
            learners = [L] * 3
            testers = [T] * 2
        elif self.role == 'spies':
            learners = [L] * 2
            testers = [T] * 3

        players = learners + testers
        
        for _ in range(n_games):
            winner = simulate_game(players, verbose=False)
            if winner == self.role:
                win += 1
            

        return win / n_games

    def get_archive(self):
        """
        Return a **list** of genome arrays currently stored in the archive
       
        """
        return [ind.genes for ind in self.members]
    
    def export(self, csv_path: str):
        """
        Write the current archive to `csv_path`.

        Each row = one Individual with all its statistics and genes.
        """
        rows = []
        for ind in self.members:
            row = {
                "generation": ind.generation,
                "wins"      : ind.wins,
                "opponents" : ind.opponents,
                "a_fitness" : ind.a_fitness,
                "i_fitness" : ind.i_fitness,          # optional but nice
                "won_ids"   : json.dumps(ind.won_ids)
            }
            for i, g in enumerate(ind.genes):
                row[f"gene_{i}"] = g
            rows.append(row)

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[export] wrote {len(rows)} rows → {csv_path}")

class Individual:
    def __init__(self, genes: np.ndarray, generation: int, fitness: float = 0.0):
        self.genes = genes
        self.generation = generation        # Used as unique ID
        self.i_fitness = fitness            # Fitness recorded at insertion
        self.a_fitness: float = 0.0         # Fitness recored after tested against the whole archaive
        self.beaten_ids: Set[int] = set()   # Set containing all individuals beaten from opposite

    def __repr__(self):
        return (f"Individual(generation={self.generation}, "
                f"i_fitness={self.i_fitness:.3f}, "
                f"a_fitness={self.a_fitness:.3f}, "
                f"beaten={sorted(self.beaten_ids)})")

class CompositeArchive():
    def __init__(self, max_size):
        self.resistance_archive: List[Individual] = []
        self.spies_archive: List[Individual] = []
        self.res_coverage: Set[int] = set()
        self.spy_coverage: Set[int] = set()
        self.threshold = 0.6
        self.all_spy_generations_seen = set()
        self.all_res_generations_seen = set()
        self.max_size = max_size

    def seed(self, ind: Individual, role: str):
        ind.beaten_ids.add(ind.generation)

        if role == 'resistance':
            self.resistance_archive.append(ind)
            self.res_coverage.add(ind.generation)
            self.all_res_generations_seen.add(ind.generation)
        elif role == 'spies':
            self.spies_archive.append(ind)
            self.spy_coverage.add(ind.generation)
            self.all_spy_generations_seen.add(ind.generation)
        else:
            raise ValueError("Role must be 'resistance' or 'spies'")
        
    def remove_redundant_individuals(self, role: str):
        archive = self.resistance_archive if role == 'resistance' else self.spies_archive
        new_archive = []

        for i, ind in enumerate(archive):
            is_redundant = False
            for j, other in enumerate(archive):
                if i == j:
                    continue
                if (
                    ind.beaten_ids.issubset(other.beaten_ids)
                    and self.is_similar(ind.genes, other.genes)
                    and ind.a_fitness <= other.a_fitness
                ):
                    is_redundant = True
                    print(f"[{role.upper()} Redundant Individual Found]")
                    break
            if not is_redundant:
                new_archive.append(ind)

        if role == 'resistance':
            self.resistance_archive = new_archive
        else:
            self.spies_archive = new_archive

    def dominates(self, a: set, b: set) -> bool:
        return a > b

    def is_similar(self, genes_a: np.ndarray, genes_b: np.ndarray, percent: float = 0.25) -> bool:
        dim = genes_a.shape[0]

        if dim == 14:
            max_distance = 7.48  # √(14 × 2²)
        elif dim == 10:
            max_distance = 3.16  # √(10 × 1²)
        else:
            raise ValueError(f"Unsupported gene dimension: {dim}")

        threshold = percent * max_distance
        return np.linalg.norm(genes_a - genes_b) < threshold

    def expands_coverage(self, beaten_ids: set, coverage_set: set) -> bool:
        return not beaten_ids.issubset(coverage_set)

    def consider_insertion(self, candidate: Individual, archive: list, coverage_set: set, role: str) -> bool:

        if len(archive) < self.max_size:
            archive.append(candidate)
            return

        for i, archived in enumerate(archive):

            if self.dominates(candidate.beaten_ids, archived.beaten_ids):
                print(f"✅ [{role.upper()}] Candidate dominates archived individual at index {i}")
                print(f"🗑️ [{role.upper()}] Replacing individual at index {i}: Beaten IDs = {archive[i].beaten_ids}")
                archive[i] = candidate
                coverage_set |= candidate.beaten_ids

                if role == "resistance":
                    self.all_res_generations_seen.add(candidate.generation)
                elif role == "spies":
                    self.all_spy_generations_seen.add(candidate.generation)

                return True

            if candidate.beaten_ids == archived.beaten_ids:
                dist = np.linalg.norm(candidate.genes - archived.genes)
                print(f"🔍 [{role.upper()}] Same beaten_ids, Euclidean distance: {dist:.3f}")
                if not self.is_similar(candidate.genes, archived.genes):
                    print(f"✅ [{role.upper()}] Replacing archived at {i} due to diversity")
                    archive[i] = candidate

                    if role == "resistance":
                        self.all_res_generations_seen.add(candidate.generation)
                    elif role == "spies":
                        self.all_spy_generations_seen.add(candidate.generation)

                elif candidate.a_fitness > archived.a_fitness:
                    print(f"🔁 Replacing similar individual at index {i} due to better a_fitness: {candidate.a_fitness:.3f} > {archived.a_fitness:.3f}")
                    archive[i] = candidate

                    if role == "resistance":
                        self.all_res_generations_seen.add(candidate.generation)
                    elif role == "spies":
                        self.all_spy_generations_seen.add(candidate.generation)
                else:
                    print(f"❌ [{role.upper()}] Candidate too similar and inferior, rejected")
                return False

        if self.expands_coverage(candidate.beaten_ids, coverage_set):
            print(f"✅ [{role.upper()}] Candidate expands coverage")
            archive.append(candidate)

            if role == "resistance":
                self.all_res_generations_seen.add(candidate.generation)
            elif role == "spies":
                self.all_spy_generations_seen.add(candidate.generation)

            coverage_set |= candidate.beaten_ids
            return True

        print(f"❌ [{role.upper()}] Candidate rejected — no dominance, no coverage expansion")
        return False

    def play_and_evaluate(self, ind: Individual, opponents: List[Individual], role: str, threshold=0.6, n_games: int=100):
        beaten_ids = set()
        total_win_rate = 0

        for opponent in opponents:
            win_count = 0

            if role == 'resistance':
                players = [ind.genes] * 3 + [opponent.genes] * 2
            elif role == 'spies':
                players = [ind.genes] * 2 + [opponent.genes] * 3
            else:
                raise ValueError("Role must be 'resistance' or 'spies'")

            for _ in range(n_games):
                result = simulate_game(list_of_players=players)

                if result == role:
                    win_count += 1

            win_rate = win_count / n_games

            if win_rate > threshold:
                beaten_ids.add(opponent.generation)

            total_win_rate += win_rate

        ind.a_fitness = total_win_rate / len(opponents)
        ind.beaten_ids = beaten_ids

    def get_resistance_archive(self):
        """
        Return a **list** of genome arrays currently stored in the Resistance archive
       
        """
        return [ind.genes for ind in self.resistance_archive]
    
    def get_spies_archive(self):
        """
        Return a **list** of genome arrays currently stored in the Spies archive
       
        """
        return [ind.genes for ind in self.spies_archive]
    
    def export_resistance_archive(self, csv_path: str):
        """
        Write the current archive to `csv_path`.

        Each row = one Individual with all its statistics and genes.
        """
        rows = []
        for ind in self.resistance_archive:
            row = {
                "generation"   : ind.generation,
                "a_fitness"    : ind.a_fitness,
                "i_fitness"    : ind.i_fitness,    
                "beaten_ids"   : json.dumps(list(ind.beaten_ids))
            }
            for i, g in enumerate(ind.genes):
                row[f"gene_{i}"] = g
            rows.append(row)

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[export] wrote {len(rows)} rows → {csv_path}")

    def export_spies_archive(self, csv_path: str):
        """
        Write the current archive to `csv_path`.

        Each row = one Individual with all its statistics and genes.
        """
        rows = []
        for ind in self.spies_archive:
            row = {
                "generation"   : ind.generation,
                "a_fitness"    : ind.a_fitness,
                "i_fitness"    : ind.i_fitness,    
                "beaten_ids"   : json.dumps(list(ind.beaten_ids))
            }
            for i, g in enumerate(ind.genes):
                row[f"gene_{i}"] = g
            rows.append(row)

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[export] wrote {len(rows)} rows → {csv_path}")