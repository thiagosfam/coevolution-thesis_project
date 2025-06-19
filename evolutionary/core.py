import numpy as np
import sys
import os
import random
from numba import jit
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from engine.game import simulate_game

class EvolutionaryAlgorithm:
    def __init__(self):

        self.each_gen_best = False
        self.population = []
        self.fitness = []  # Add a separate list for fitness values

    def initialize_population(self, population_size: int, num_genes: int, value_range: tuple):
        # Initialize population with random genes

        for _ in range(population_size):
            individual = np.random.uniform(value_range[0], value_range[1], num_genes)
            self.population.append(individual)

        return self.population
    
    def evaluate_individual_HoF(self, individual_genes, role, opponent_pool_hof, n_games=100):
        """
        Evaluate a single individual by playing multiple games against Hall of Fame opponents.
        
        Args:
            individual_genes: numpy array of genes
            role: 'resistance' or 'spy'
            opponent_pool: list of individuals from the opposite role
            opponent_pool_hof: list of individuals from the opposite role
            n_games: number of games to play for evaluation (default: 100)
            
        Returns:
            float: average fitness score (win rate)
        """
        # Ensure we have enough players
        if role == 'resistance':
            n_opponents = 2
            n_coplayers = 2
        else:  # spy
            n_opponents = 3
            n_coplayers = 1

            
        total_fitness = 0.0
        games_played = 0
        
        
        for _ in range(n_games):
            # Select opponents from opponent Hall of Fame
            selected_opponents = random.choices(opponent_pool_hof, k=n_opponents)

            # Select co-players from co-player Hall of Fame
            coplayers = [individual_genes] * n_coplayers
            
            # Create list of all players' genes
            all_genes = []
            
            # Add the individual being evaluated
            all_genes.append(individual_genes)
            
            # Add co-players 
            for coplayer in coplayers:
                all_genes.append(coplayer)
                
            # Add opponents
            for opponent in selected_opponents:
                all_genes.append(opponent)
            
            # Play the game
            result = simulate_game(all_genes)
            
            # Update fitness based on role and result
            if role == 'resistance':
                if result == 'resistance':
                    total_fitness += 1.0
            else:  # spy
                if result == 'spies':
                    total_fitness += 1.0
                    
            games_played += 1
            
        return total_fitness / games_played if games_played > 0 else 0.0
    
    def _evaluate_individual(self, individual, role, opponent_pool, n_games=50):
        """
        Evaluate a single individual by playing multiple games against Hall of Fame opponents.
        
        Args:
            individual_genes: numpy array of genes
            role: 'resistance' or 'spy'
            coplayer_pool: list of individuals from the same role
            opponent_pool: list of individuals from the opposite role
            n_games: number of games to play for evaluation (default: 100)
            
        Returns:
            float: average fitness score (win rate)
        """
        # Ensure we have enough players
        if role == 'resistance':
            n_players= 3
            n_opponents = 2
        else:  # spy
            n_players = 2
            n_opponents = 3
            
        total_fitness = 0.0
        games_played = 0
        
        
        for _ in range(n_games):
            # Co-players are identical to individual being evaluated
            coplayers = [individual] * n_players
            opponents = [random.choice(opponent_pool)] * n_opponents
            
            # Create list of all players' genes
            all_players = coplayers + opponents
            
            # Setup and play the game
            result = simulate_game(all_players)
            
            # Update fitness based on role and result
            if role == 'resistance':
                if result == 'resistance':
                    total_fitness += 1.0
            else:  # spy
                if result == 'spies':
                    total_fitness += 1.0
                    
            games_played += 1
            
        return total_fitness / games_played if games_played > 0 else 0.0
    
    
    def evaluate_individual(self, individual, role, coplayers, opponents, n_games=100):
        """
        Evaluate a single individual by playing multiple games against Hall of Fame opponents.
        
        Args:
            individual_genes: numpy array of genes
            role: 'resistance' or 'spy'
            coplayer_pool: list of individuals from the same role
            opponent_pool: list of individuals from the opposite role
            n_games: number of games to play for evaluation (default: 100)
            
        Returns:
            float: average fitness score (win rate)
        """
        # Ensure we have enough players
        if role == 'resistance':
            if len(coplayers) < 2:
                raise ValueError("Need at least 2 co-players for resistance evaluation")
            n_coplayers = 2
            n_opponents = 2
        else:  # spy
            if len(coplayers) < 1:
                raise ValueError("Need at least 1 co-player for spy evaluation")
            n_coplayers = 1
            n_opponents = 3
            
        total_fitness = 0.0
        games_played = 0
        
        
        for _ in range(n_games):
            # Select opponents
            if len(coplayers) < n_coplayers:
                 raise ValueError(f"Not enough opponents in evaluation pool ({len(opponents)}) to select {n_opponents}")
            
            # Select opponents
            if len(opponents) < n_opponents:
                 raise ValueError(f"Not enough opponents in evaluation pool ({len(opponents)}) to select {n_opponents}")

            
            # Create list of all players' genes
            all_players = []
            
            # Add the individual being evaluated
            all_players.append(individual)
            
            # Add co-players (extract genes from tuples if needed)
            for coplayer in coplayers:
                all_players.append(coplayer)

            
            # Add opponents (extract genes from tuples if needed)
            for opponent in opponents:
                all_players.append(opponent)
            # Setup and play the game
            result = simulate_game(all_players)
            
            # Update fitness based on role and result
            if role == 'resistance':
                if result == 'resistance':
                    total_fitness += 1.0
            else:  # spy
                if result == 'spies':
                    total_fitness += 1.0
                    
            games_played += 1
            
        return total_fitness / games_played if games_played > 0 else 0.0
    
    def evaluate_population_HoF(self, population: list, opponent_pool_hof: list, role: str):
        """
        Evaluate a population by playing games.
        
        Args:
            population: list of individuals to evaluate
            opponent_pool_hof: list of individuals from the opposite role
            role: 'resistance' or 'spy'
            
        Returns:
            list of tuples (individual, fitness)
        """ 
        evaluated = []
        for individual in population:
            # Extract genes if individual is a tuple
            genes = individual[0] if isinstance(individual, tuple) else individual
            fitness = self.evaluate_individual_HoF(genes, role, opponent_pool_hof)
            evaluated.append((genes, fitness))
        return evaluated
    
    def evaluate_population(self, population: list, coplayer_pool: list, opponent_pool: list, role: str):
        """
        Evaluate a population by playing games.
        
        Args:
            population: list of individuals to evaluate
            coplayer_pool: list of individuals from the same role
            opponent_pool: list of individuals from the opposite role
            role: 'resistance' or 'spy'
            
        Returns:
            list of tuples (individual, fitness)
        """
        evaluated = []
        for individual in population:
            # Extract genes if individual is a tuple
            genes = individual[0] if isinstance(individual, tuple) else individual
            fitness = self.evaluate_individual(genes, role, coplayer_pool, opponent_pool)
            evaluated.append((genes, fitness))
        return evaluated
        
    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray, crossover_probability: float = 0.1):
        """
        Performs uniform crossover between two parents with a given probability.
        For each gene position, randomly selects the gene from either parent with equal probability.
        
        Args:
            parent1: First parent's genes
            parent2: Second parent's genes
            crossover_probability: Probability of performing crossover (default 0.1 or 10%)
            
        Returns:
            tuple: Two offspring created by uniform crossover or copies of parents if crossover doesn't occur
        """
        # Check if crossover should occur based on probability
        if np.random.random() > crossover_probability:
            # If no crossover, return copies of parents
            return np.copy(parent1), np.copy(parent2)
        
        # Ensure parents have the same length
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same number of genes")
        
        # Create masks for random selection
        mask = np.random.random(len(parent1)) < 0.5
        
        # Create offspring using the masks
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        
        return offspring1, offspring2
        
    def gaussian_mutation(self, individual: np.ndarray, mutation_probability: float = 1.0, 
                         sigma: float = 0.1, value_range: tuple = None):
        """
        Applies Gaussian mutation to an individual with a given probability.
        Each gene has a chance to be mutated by adding a random value from a Gaussian distribution.
        Values can be clamped to a specified range.
        
        Args:
            individual: The individual to mutate
            mutation_probability: Probability of mutating each gene (default 0.1 or 10%)
            sigma: Standard deviation of the Gaussian distribution (default 0.1)
            value_range: Tuple of (min_value, max_value) to clamp genes to (default None, no clamping)
            
        Returns:
            np.ndarray: The mutated individual
        """
        # Create a copy of the individual to avoid modifying the original
        mutated = np.copy(individual)
        
        # Create a mask for genes to mutate based on probability
        mutation_mask = np.random.random(len(individual)) < mutation_probability
        
        # Generate Gaussian noise for all genes
        noise = np.random.normal(0, sigma, len(individual))
        
        # Apply noise only to genes selected by the mask
        mutated[mutation_mask] += noise[mutation_mask]
        
        # Clamp values if a range is specified
        if value_range is not None:
            min_val, max_val = value_range
            mutated = np.clip(mutated, min_val, max_val)
        
        return mutated
        
    def parent_selection(self, population: list):
        """
        Selects 2 parents randomly and uniformly from the population.
        
        Args:
            population_with_fitness: List of tuples (individual, fitness)
            
        Returns:
            tuple: Two selected parents
        """
        # Randomly select 2 individuals without replacement
        selected_pairs = random.sample(population, 2)
        
        # Extract just the individuals (parents)
        parent1, parent2 = selected_pairs[0], selected_pairs[1]
        
        return parent1, parent2
        
    def survival_selection(self, population_with_fitness: list, mu: int):
        """
        Selects the mu best individuals from a population based on fitness.
        
        Args:
            population_with_fitness: List of tuples (individual, fitness)
            mu: Number of individuals to select (survivors)
            
        Returns:
            list: Selected individuals with their fitness (the mu best)
        """
        # Sort the population by fitness (descending)
        sorted_population = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
        
        # Select the mu best individuals
        survivors = sorted_population[:mu]
        
        return survivors
        
    def evolve(self, population: list, opponent_pool: list, role: str, 
              num_gen: int = 100, mu: int = 21, lambda_: int = 150,
              p_cross: float = 0.1, std: float = 0.1, hof_size: int = 100):
        """
        Performs the evolutionary loop for a specified number of generations.
        
        Args:
            population: Initial population
            opponent_pool: List of potential opponents
            role: 'resistance' or 'spy'
            num_gen: Number of generations to evolve
            mu: Population size (number of individuals to select for the next generation)
            lambda_: Offspring size (number of offspring to create)
            p_cross: Crossover probability
            std: Standard deviation for Gaussian mutation
            hof_size: Size of the Hall of Fame
            
        Returns:
            tuple: (final population, Hall of Fame, statistics)
        """
        self.HoF_size = hof_size
        HoF = []
        stats = {
            'min_fitness': [],
            'max_fitness': [],
            'avg_fitness': [],
            'std_fitness': [],
            'best_individuals': []
        }
        
        value_range = (0, 1) if role == 'spy' else (-1, 1)
        population_with_fitness = self.evaluate_population(population, population, opponent_pool, role)
        
        print(f"Starting evolution for {role} - {num_gen} generations")
        for generation in range(num_gen):
            print(f"Generation {generation+1}/{num_gen} - Best fitness: {stats['max_fitness'][-1] if stats['max_fitness'] else 'N/A'}")
            
            offspring = []
            for _ in range(lambda_ // 2):
                parent1, parent2 = self.parent_selection(population_with_fitness)
                offspring1, offspring2 = self.uniform_crossover(parent1, parent2, p_cross)
                offspring1 = self.gaussian_mutation(offspring1, 0.1, std, value_range)
                offspring2 = self.gaussian_mutation(offspring2, 0.1, std, value_range)
                offspring.append(offspring1)
                offspring.append(offspring2)
            
            offspring_with_fitness = self.evaluate_population(offspring, offspring, opponent_pool, role)
            population = self.survival_selection(offspring_with_fitness, mu)
            
            fitness_values = [fit for _, fit in population]
            stats['min_fitness'].append(min(fitness_values))
            stats['max_fitness'].append(max(fitness_values))
            stats['avg_fitness'].append(sum(fitness_values) / len(fitness_values))
            stats['std_fitness'].append(np.std(fitness_values) if len(fitness_values) > 1 else 0)
            
            best_individual, best_fitness = max(population, key=lambda x: x[1])
            stats['best_individuals'].append((best_individual, best_fitness))
            
            if len(HoF) < self.HoF_size:
                HoF.append((best_individual, best_fitness))
                HoF.sort(key=lambda x: x[1], reverse=True)
            elif best_fitness > HoF[-1][1]:
                HoF[-1] = (best_individual, best_fitness)
                HoF.sort(key=lambda x: x[1], reverse=True)
            
            population_with_fitness = population
        
        print(f"\nEvolution completed for {role}")
        print(f"Final best fitness: {best_fitness:.4f}")
        
        return population, HoF, stats

#################################### TESTING #####################################

if __name__ == "__main__":
    # Create an instance of the EvolutionaryAlgorithm
    ea = EvolutionaryAlgorithm()
    
    # Create populations
    print("Initializing populations...")
    spies = []
    for _ in range(50):
        individual = np.random.uniform(0, 1, 10)
        spies.append(individual)
    
    resistance = ea.initialize_population(
        population_size=21,
        num_genes=14,
        value_range=(-1, 1)
    )
    
    print(f"Starting evolution with {len(spies)} spies and {len(resistance)} resistance fighters")
    
    # Evolve the resistance population
    final_resistance, resistance_HoF, resistance_stats = ea.evolve(
        population=resistance,
        opponent_pool=spies,
        role='resistance',
        num_gen=200,
        mu=21,
        lambda_=150,
        p_cross=0.1,
        std=0.1,
        hof_size=100
    )
    
    # Create results directory if needed
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Export results
    print("\nExporting results...")
    
    # Export final population
    final_population_data = []
    for i, (individual, fitness) in enumerate(final_resistance):
        row = {"id": i, "fitness": fitness}
        for j, gene in enumerate(individual):
            row[f"gene_{j}"] = gene
        final_population_data.append(row)
    
    df = pd.DataFrame(final_population_data)
    csv_path = os.path.join(results_dir, "final_resistance_population.csv")
    df.to_csv(csv_path, index=False)
    
    # Export Hall of Fame
    hof_data = []
    for i, (individual, fitness) in enumerate(resistance_HoF):
        row = {"id": i, "fitness": fitness}
        for j, gene in enumerate(individual):
            row[f"gene_{j}"] = gene
        hof_data.append(row)
    
    df_hof = pd.DataFrame(hof_data)
    hof_csv_path = os.path.join(results_dir, "resistance_hall_of_fame.csv")
    df_hof.to_csv(hof_csv_path, index=False)
    
    # Create visualization
    generations = range(1, len(resistance_stats['avg_fitness']) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(generations, resistance_stats['avg_fitness'], 'b-', linewidth=2, label='Average Fitness')
    plt.plot(generations, resistance_stats['max_fitness'], 'b-', alpha=0.5, linewidth=1, label='Max Fitness')
    plt.fill_between(
        generations,
        [avg - std for avg, std in zip(resistance_stats['avg_fitness'], resistance_stats['std_fitness'])],
        [avg + std for avg, std in zip(resistance_stats['avg_fitness'], resistance_stats['std_fitness'])],
        color='blue', alpha=0.2, label='Standard Deviation'
    )
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Performance')
    
    plt.title('Evolution of Resistance Fighters', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plot_path = os.path.join(results_dir, "resistance_evolution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print("Results exported successfully")

