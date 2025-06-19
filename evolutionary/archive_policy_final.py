import random
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Optional
from engine.game import simulate_game
import json
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

class Individual:
    def __init__(self, genes: np.array, fitness: float, role: str, generation:int):
        self.genes = genes
        self.insertion_fitness = fitness
        self.role = role  # 'resistance' or 'spy'
        self.generation = generation
        self.novelty_score = None  # To be computed later
        self.archive_fitness = None  # Used in cross-evaluation
        self.num_wins = int
        self.num_games = int

    def __repr__(self):
        return (f"Individual(generation={self.generation}, "
                f"insertion_fitness={self.insertion_fitness:.3f}, "
                f"archive_fitness={self.archive_fitness:.3f}, "
                f"novelty_score={self.novelty_score})")

class ArchiveManager:
    def __init__(self, max_size):
        self.resistance_archive: List[Individual] = []
        self.spy_archive: List[Individual] = []
        self.max_size = max_size

    def add(self, individual: Individual):
        assert isinstance(individual, Individual), "Must be an Individual object"
        assert individual.role in ["resistance", "spy"], "Invalid role"
        if individual.role == "resistance":
            self.resistance_archive.append(individual)
        else:
            self.spy_archive.append(individual)

    def get_archive(self, role: str) -> List[Individual]:
        if role == "resistance":
            return self.resistance_archive
        elif role == "spy":
            return self.spy_archive
        else:
            raise ValueError("Role must be 'resistance' or 'spy'")

    def cross_evaluate(self):
        n_games = 10

        # Reset stats
        for ind in self.resistance_archive + self.spy_archive:
            ind.num_wins = 0
            ind.num_games = 0

        # Evaluate each Resistance agent against each Spy agent
        for res_ind in self.resistance_archive:
            for spy_ind in self.spy_archive:

                players = [res_ind.genes] * 3 + [spy_ind.genes] * 2

                for _ in range(n_games):
                    winer = simulate_game(list_of_players=players)
                    
                    # Update both agents
                    res_ind.num_games += 1
                    spy_ind.num_games += 1

                    if winer == 'resistance':
                        res_ind.num_wins += 1
                    elif winer == "spies":
                        spy_ind.num_wins += 1

            

        # Calculate and update archive fitness for each individual
        for ind in self.resistance_archive + self.spy_archive:
            if ind.num_games > 0:
                ind.archive_fitness = ind.num_wins / ind.num_games
            else:
                ind.archive_fitness = None  # No valid games played

    def _multiobjective_prune(self, individuals: List[Individual], alpha=1.0, beta=1.0) -> List[Individual]:
        scored = []
        for ind in individuals:
            if ind.archive_fitness is not None and ind.novelty_score is not None:
                score = alpha * ind.archive_fitness + beta * ind.novelty_score
                scored.append((score, ind))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [ind for _, ind in scored[:self.max_size]]

    def _clustering_prune(self, individuals: List[Individual], k: int = 3, keep_novel: int = 1) -> List[Individual]:
        if len(individuals) <= self.max_size:
            return individuals

        gene_matrix = np.stack([ind.genes for ind in individuals])
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(gene_matrix)

        clusters = {i: [] for i in range(k)}
        for ind, label in zip(individuals, labels):
            clusters[label].append(ind)

        selected = []
        for group in clusters.values():
            best = max(group, key=lambda ind: ind.archive_fitness if ind.archive_fitness is not None else -1)
            selected.append(best)

        if keep_novel > 0:
            already_selected = set(selected)
            novel_candidates = [ind for ind in individuals if ind not in already_selected]
            sorted_by_novelty = sorted(
                novel_candidates,
                key=lambda ind: ind.novelty_score if ind.novelty_score is not None else 0,
                reverse=True
            )
            selected.extend(sorted_by_novelty[:keep_novel])

        return selected[:self.max_archive_size]
    
    def compute_novelty(self, role: str, k: int = 3):
        """
        Computes a novelty score for each individual based on the average distance
        to their k nearest neighbors in gene space.
        """
        archive = self.get_archive(role)

        if len(archive) <= k:
            for ind in archive:
                ind.novelty_score = 0.0
            return
        
        # Stack all genes into a matrix
        gene_matrix = np.stack([ind.genes for ind in archive])

        # Compute pairwise distances
        distance_matrix = pairwise_distances(gene_matrix, metric="euclidean")

        # Compute novelty as average distance to k nearest neighbors (excluding self)
        for i, ind in enumerate(archive):
            sorted_distances = np.sort(distance_matrix[i])
            ind.novelty_score = np.mean(sorted_distances[1:k+1])  # skip self

    def prune(self, method: str = "multiobjective", role: str = "resistance", **kwargs):
        archive = self.get_archive(role)
        if method == "multiobjective":
            pruned = self._multiobjective_prune(archive, **kwargs)
        elif method == "clustering":
            pruned = self._clustering_prune(archive, **kwargs)
        else:
            raise ValueError("Unknown pruning method")

        # Update the archive
        if role == "resistance":
            self.resistance_archive = pruned
        elif role == "spy":
            self.spy_archive = pruned

        return pruned
    
    def get_genomes(self, role: str) -> List[np.array]:
        """
        Return a **list** of genome arrays currently stored in the specified archive.
       
        """
        return [ind.genes for ind in self.get_archive(role=role)]
    
    def export_archive(self, role: str, csv_path: str):
        """
        Export archive for the given role ('resistance' or 'spy') to a CSV file.
        Each row includes metadata + all gene values.
        """
        archive = self.get_archive(role)
        rows = []

        for ind in archive:
            row = {
                "generation": ind.generation,
                "insertion_fitness": ind.insertion_fitness,
                "archive_fitness": ind.archive_fitness,
                "novelty_score": ind.novelty_score,
            }

            for i, g in enumerate(ind.genes):
                row[f"gene_{i}"] = g

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[export] wrote {len(df)} rows → {csv_path}")


def cluster_and_select_best(individuals: List[Individual], k: int = 3) -> List[Individual]:
    """
    Cluster individuals and select the best one from each cluster based on archive_fitness.
    Also returns cluster assignments.
    """
    if len(individuals) <= k:
        return individuals  # Not enough to cluster; keep all

    gene_matrix = np.stack([ind.genes for ind in individuals])
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(gene_matrix)

    clusters = {i: [] for i in range(k)}
    for ind, label in zip(individuals, labels):
        clusters[label].append(ind)

    best_per_cluster = []
    for group in clusters.values():
        best = max(group, key=lambda ind: ind.archive_fitness if ind.archive_fitness is not None else -1)
        best_per_cluster.append(best)

    return best_per_cluster

# IMPLEMENTATION A: Multiobjective Filter (fitness + novelty)

def multiobjective_select(individuals: List[Individual], top_n: int = 3, alpha: float = 1.0, beta: float = 1.0) -> List[Individual]:
    """
    Select top N individuals by combining archive_fitness and novelty_score into a weighted score.
    """
    scored = []
    for ind in individuals:
        if ind.archive_fitness is not None and ind.novelty_score is not None:
            score = alpha * ind.archive_fitness + beta * ind.novelty_score
            scored.append((score, ind))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return [ind for _, ind in scored[:top_n]]


# IMPLEMENTATION B: Clustering + Best Per Cluster + Optional Novelty

def clustering_select_with_novelty(
    individuals: List[Individual],
    k: int = 3,
    keep_novel: int = 1
) -> List[Individual]:
    """
    Select the best individual per cluster (by archive_fitness).
    Optionally keep additional 'novel' individuals not already selected.
    """
    if len(individuals) <= k:
        return individuals

    gene_matrix = np.stack([ind.genes for ind in individuals])
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(gene_matrix)

    clusters = {i: [] for i in range(k)}
    for ind, label in zip(individuals, labels):
        clusters[label].append(ind)

    selected = []
    for group in clusters.values():
        best = max(group, key=lambda ind: ind.archive_fitness if ind.archive_fitness is not None else -1)
        selected.append(best)

    if keep_novel > 0:
        # Sort by novelty and add top 'novel' individuals not already selected
        already_selected = set(selected)
        sorted_by_novelty = sorted(
            [ind for ind in individuals if ind not in already_selected],
            key=lambda ind: ind.novelty_score if ind.novelty_score is not None else 0,
            reverse=True
        )
        selected.extend(sorted_by_novelty[:keep_novel])

    return selected

"###################### TESTING SECTION ######################"

"""
# Re-instantiate the archive manager with a size cap
archive = ArchiveManager(max_size=3)

# Generate sample Resistance and Spy individuals with dummy values
np.random.seed(42)  # for reproducibility

# Resistance: 14 genes in [-1, 1]
for i in range(6):
    genes = np.random.uniform(-1, 1, 14)
    fitness = np.random.uniform(0.6, 0.9)
    archive.add(Individual(generation=i, genes=genes, fitness=fitness, role="resistance"))

# Spy: 10 genes in [0, 1]
for i in range(6):
    genes = np.random.uniform(0, 1, 10)
    fitness = np.random.uniform(0.5, 0.8)
    archive.add(Individual(generation=i, genes=genes, fitness=fitness, role="spy"))

# Assign dummy archive fitness and novelty for pruning
for ind in archive.get_archive("resistance") + archive.get_archive("spy"):
    ind.archive_fitness = np.random.uniform(0.4, 0.9)
    ind.novelty_score = np.random.uniform(1.0, 5.0)

# Prune resistance and spy archives using both methods
res_mo = archive.prune(method="multiobjective", role="resistance", alpha=1.0, beta=1.0)
spy_mo = archive.prune(method="multiobjective", role="spy", alpha=1.0, beta=1.0)

res_cluster = archive.prune(method="clustering", role="resistance", k=2, keep_novel=1)
spy_cluster = archive.prune(method="clustering", role="spy", k=2, keep_novel=1)

# Show the final individuals selected
res_mo_summary = [(round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in res_mo]
spy_mo_summary = [(round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in spy_mo]
res_cluster_summary = [(round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in res_cluster]
spy_cluster_summary = [(round(ind.archive_fitness, 2), round(ind.novelty_score, 2)) for ind in spy_cluster]

print(res_mo_summary) 
print(spy_mo_summary) 
print(res_cluster_summary) 
print(spy_cluster_summary)

"""