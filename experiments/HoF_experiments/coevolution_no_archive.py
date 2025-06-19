import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import matplotlib.ticker as ticker
from evolutionary.core import EvolutionaryAlgorithm

def co_evolve(resistance_ea, spy_ea, 
              num_gen=200, mu=21, lambda_=150,
              p_cross=0.1, std=0.1, hof_size=None):
    """
    Co-evolves two populations: resistance fighters and spies.
    The standard deviation for spy mutations decreases by 0.5% each generation.
    
    Args:
        resistance_ea: EvolutionaryAlgorithm instance for resistance fighters
        spy_ea: EvolutionaryAlgorithm instance for spies
        num_gen: Number of generations to evolve
        mu: Population size (number of individuals to select for the next generation)
        lambda_: Offspring size (number of offspring to create)
        p_cross: Crossover probability
        std: Initial standard deviation for Gaussian mutation
        hof_size: Size of the Hall of Fame
        
    Returns:
        tuple: (final_resistance, final_spies, resistance_HoF, spy_HoF, resistance_stats, spy_stats, resistance_hof, spy_hof)
    """
    # Initialize statistics tracking for both populations
    resistance_stats = {
        'max_fitness': [],
        'avg_fitness': [],
        'std_fitness': [],
        'best_individuals': []
    }
    
    spy_stats = {
        'max_fitness': [],
        'avg_fitness': [],
        'std_fitness': [],
        'best_individuals': [],
    }
    
    resistance_hof = {
        'generation': [],
        'fitness': [],
        'genes': []
    }
    spy_hof = {
        'generation': [],
        'fitness': [],
        'genes': []
    }
    
    # Get initial populations
    resistance_pop = resistance_ea.population
    spy_pop = spy_ea.population

     # 1. Evaluate resistance population against the first spy population
    print("Evaluating initial resistance population...")
    resistance_with_fitness = []

    for ind in resistance_pop:
        fitness = resistance_ea._evaluate_individual(individual=ind, 
                                                         role='resistance', 
                                                         opponent_pool=spy_pop)
        resistance_with_fitness.append((ind, fitness))

        
    # 2. Evaluate spy population against  the first resistance population
    print("Evaluating initial spy population...")
    spy_with_fitness = []

    for ind in spy_pop:
        fitness = spy_ea._evaluate_individual(individual=ind, 
                                                         role='spy', 
                                                         opponent_pool=resistance_pop)
        spy_with_fitness.append((ind, fitness))

    # Initialize spy mutation standard deviation
    spy_std = std
    
    # Main co-evolutionary loop
    for generation in range(num_gen):
        print(f"Generation {generation+1}/{num_gen}")
        
        
        # 3. Create offspring for resistance population
        print("Creating resistance offspring...")
        resistance_offspring = []
        for _ in range(lambda_ // 2):  # Divide by 2 because crossover creates 2 offspring
            # Select parents
            parent1, parent2 = resistance_ea.parent_selection(resistance_pop)
            
            # Perform crossover
            offspring1, offspring2 = resistance_ea.uniform_crossover(parent1, parent2, p_cross)
            
            # Apply mutation
            offspring1 = resistance_ea.gaussian_mutation(offspring1, 1.0, std, (-1, 1))
            offspring2 = resistance_ea.gaussian_mutation(offspring2, 1.0, std, (-1, 1))
            
            # Add to offspring
            resistance_offspring.append(offspring1)
            resistance_offspring.append(offspring2)
        
        # 4. Create offspring for spy population
        print("Creating spy offspring...")
        spy_offspring = []
        for _ in range(lambda_ // 2):  # Divide by 2 because crossover creates 2 offspring
            # Select parents
            parent1, parent2 = spy_ea.parent_selection(spy_pop)
            
            # Perform crossover
            offspring1, offspring2 = spy_ea.uniform_crossover(parent1, parent2, p_cross)
            
            # Apply mutation with decaying standard deviation
            offspring1 = spy_ea.gaussian_mutation(offspring1, 1.0, spy_std, (0, 1))
            offspring2 = spy_ea.gaussian_mutation(offspring2, 1.0, spy_std, (0, 1))
            
            # Add to offspring
            spy_offspring.append(offspring1)
            spy_offspring.append(offspring2)
        

        # 5. Evaluate resistance offspring against spy population
        print("Evaluating resistance offspring...")
        resistance_offspring_with_fitness = []

        for ind in resistance_offspring:
            fitness = resistance_ea._evaluate_individual(individual=ind, 
                                                         role='resistance', 
                                                         opponent_pool=spy_pop)
            resistance_offspring_with_fitness.append((ind, fitness))


        # 6. Evaluate spy offspring against resistance population
        print("Evaluating spy offspring...")
        spy_offspring_with_fitness = []
            
        for ind in spy_offspring:
            fitness = spy_ea._evaluate_individual(individual=ind, 
                                                  role='spy', 
                                                  opponent_pool=resistance_pop)
            spy_offspring_with_fitness.append((ind, fitness))

        
        # 7. Select survivors for resistance population
        resistance_survivors = resistance_ea.survival_selection(resistance_offspring_with_fitness, mu)
        # Create new population (extract just the genes)
        resistance_pop = [genes for (genes, _) in resistance_survivors]
     
        
        # 8. Select survivors for spy population
        spy_survivors = spy_ea.survival_selection(spy_offspring_with_fitness, mu)
        spy_pop = [genes for (genes, _) in spy_survivors]
        
        # 9. Calculate statistics for resistance population
        resistance_fitness_values = [fit for _, fit in resistance_survivors]
        resistance_stats['max_fitness'].append(max(resistance_fitness_values))
        resistance_stats['avg_fitness'].append(sum(resistance_fitness_values) / len(resistance_fitness_values))
        resistance_stats['std_fitness'].append(np.std(resistance_fitness_values) if len(resistance_fitness_values) > 1 else 0)
        
        # Find the best resistance individual in this generation
        best_resistance, best_resistance_fitness = max(resistance_survivors, key=lambda x: x[1])
        resistance_stats['best_individuals'].append((best_resistance, best_resistance_fitness))
        resistance_hof['generation'].append(generation)
        resistance_hof['fitness'].append(best_resistance_fitness)
        resistance_hof['genes'].append(best_resistance)
        
        # 10. Calculate statistics for spy population
        spy_fitness_values = [fit for _, fit in spy_survivors]
        spy_stats['max_fitness'].append(max(spy_fitness_values))
        spy_stats['avg_fitness'].append(sum(spy_fitness_values) / len(spy_fitness_values))
        spy_stats['std_fitness'].append(np.std(spy_fitness_values) if len(spy_fitness_values) > 1 else 0)
        
        # Find the best spy individual in this generation
        best_spy, best_spy_fitness = max(spy_survivors, key=lambda x: x[1])
        spy_stats['best_individuals'].append((best_spy, best_spy_fitness))
        spy_hof['generation'].append(generation)
        spy_hof['fitness'].append(best_spy_fitness)
        spy_hof['genes'].append(best_spy)
        
        # 11. Print statistics for this generation
        print(f"  Resistance - Max: {resistance_stats['max_fitness'][-1]:.4f}, Avg: {resistance_stats['avg_fitness'][-1]:.4f}")
        print(f"  Spies -  Max: {spy_stats['max_fitness'][-1]:.4f}, Avg: {spy_stats['avg_fitness'][-1]:.4f}")
        
        # Decrease spy mutation standard deviation by 0.5%
        spy_std *= 0.995  # Multiply by (1 - 0.005) to decrease by 0.5%
    
    # Return the final populations, Hall of Fame and statistics
    return resistance_pop, spy_pop, resistance_stats, spy_stats, resistance_hof, spy_hof

def visualize_coevolution(resistance_stats, spy_stats, save_path=None):
    """
    Creates a visualization of the co-evolution results.
    
    Args:
        resistance_stats: Statistics for the resistance population
        spy_stats: Statistics for the spy population
        save_path: Path to save the visualization (optional)
    """
    generations = range(1, len(resistance_stats['avg_fitness']) + 1)
    
    # Create figure with a single subplot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot resistance fitness
    ax1.plot(generations, resistance_stats['avg_fitness'], 'b-', linewidth=1, label='Resistance Avg Fitness')
    ax1.plot(generations, resistance_stats['max_fitness'], 'b-', alpha=0.5, linewidth=0.5, label='Resistance Max Fitness')
    
    # Plot spy fitness
    ax1.plot(generations, spy_stats['avg_fitness'], 'r-', linewidth=1, label='Spy Avg Fitness')
    ax1.plot(generations, spy_stats['max_fitness'], 'r-', alpha=0.5, linewidth=0.5, label='Spy Max Fitness')
    
    # Plot standard deviation as a shaded area
    ax1.fill_between(
        generations,
        [avg - std for avg, std in zip(resistance_stats['avg_fitness'], resistance_stats['std_fitness'])],
        [avg + std for avg, std in zip(resistance_stats['avg_fitness'], resistance_stats['std_fitness'])],
        color='blue', alpha=0.2, label='Resistance Std Dev'
    )
    
    ax1.fill_between(
        generations,
        [avg - std for avg, std in zip(spy_stats['avg_fitness'], spy_stats['std_fitness'])],
        [avg + std for avg, std in zip(spy_stats['avg_fitness'], spy_stats['std_fitness'])],
        color='red', alpha=0.2, label='Spy Std Dev'
    )
    
    
    ax1.set_title('Co-Evolution of Resistance Fighters and Spies (no Archive Evaluation)', fontsize=18)
    ax1.set_xlabel('Generation', fontsize=14)
    ax1.set_ylabel('Fitness', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Co-evolution plot saved to {save_path}")
    
    # Show the plot
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

def export_stats(resistance_stats, spy_stats, results_dir=None):
    """
    Exports the resistance and spy statistics to CSV files.
    
    Args:
        resistance_stats: Dictionary containing resistance population statistics
        spy_stats: Dictionary containing spy population statistics
        results_dir: Directory to save the results
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Export resistance statistics
    print("\nExporting resistance statistics to CSV...")
    pd.DataFrame({
        'max_fitness': resistance_stats['max_fitness'],
        'avg_fitness': resistance_stats['avg_fitness'],
        'std_fitness': resistance_stats['std_fitness']
    }).to_csv(os.path.join(results_dir, "resistance_stats_no_archive.csv"), index=False)
    print(f"Resistance statistics exported to {os.path.join(results_dir, 'resistance_stats_no_archive.csv')}")
    
    # Export spy statistics
    print("\nExporting spy statistics to CSV...")
    pd.DataFrame({
        'max_fitness': spy_stats['max_fitness'],
        'avg_fitness': spy_stats['avg_fitness'],
        'std_fitness': spy_stats['std_fitness']
    }).to_csv(os.path.join(results_dir, "spy_stats_no_archive.csv"), index=False)
    print(f"Spy statistics exported to {os.path.join(results_dir, 'spy_stats_no_archive.csv')}")

def export_populations(resistance_pop, spy_pop, results_dir=None):
    """
    Exports the final populations to CSV files.
    
    Args:
        resistance_pop: Final resistance population
        spy_pop: Final spy population
        resistance_HoF: Hall of Fame for resistance population
        spy_HoF: Hall of Fame for spy population
        results_dir: Directory to save the results
    """
    # Create a directory for results if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Export the final resistance population
    print("\nExporting final resistance population to CSV...")
    resistance_data = []
    for individual in resistance_pop:
        row = {}
        for j, gene in enumerate(individual):
            row[f'gene_{j}'] = gene
        resistance_data.append(row)
    
    df_resistance = pd.DataFrame(resistance_data)
    resistance_csv_path = os.path.join(results_dir, "coevolution_final_resistance_population_no_archive.csv")
    df_resistance.to_csv(resistance_csv_path, index=False)
    print(f"Final resistance population exported to {resistance_csv_path}")
    
    # Export the final spy population
    print("Exporting final spy population to CSV...")
    spy_data = []
    for individual in spy_pop:
        row = {}
        for j, gene in enumerate(individual):
            row[f'gene_{j}'] = gene
        spy_data.append(row)
    
    df_spy = pd.DataFrame(spy_data)
    spy_csv_path = os.path.join(results_dir, "coevolution_final_spy_population_no_archive.csv")
    df_spy.to_csv(spy_csv_path, index=False)
    print(f"Final spy population exported to {spy_csv_path}")

def export_hof(resistance_hof, spy_hof, results_dir=None):
    """
    Exports the HoF (best individual from each generation) to CSV files.
    
    Args:
        resistance_hof: Dictionary containing generation, fitness, and genes lists for resistance fighters
        spy_hof: Dictionary containing generation, fitness, and genes lists for spies
        results_dir: Directory to save the results
    """
    # Export the resistance HoF
    print("\nExporting resistance HoF to CSV...")
    resistance_data = []
    for i in range(len(resistance_hof['generation'])):
        row = {
            'generation': resistance_hof['generation'][i],
            'fitness': resistance_hof['fitness'][i]
        }
        # Add genes
        for j, gene in enumerate(resistance_hof['genes'][i]):
            row[f'gene_{j}'] = gene
        resistance_data.append(row)
    
    df_resistance = pd.DataFrame(resistance_data)
    resistance_csv_path = os.path.join(results_dir, "coevolution_resistance_hof_no_archive.csv")
    df_resistance.to_csv(resistance_csv_path, index=False)
    print(f"Resistance HoF exported to {resistance_csv_path}")
    
    # Export the spy HoF
    print("Exporting spy HoF to CSV...")
    spy_data = []
    for i in range(len(spy_hof['generation'])):
        row = {
            'generation': spy_hof['generation'][i],
            'fitness': spy_hof['fitness'][i]
        }
        # Add genes
        for j, gene in enumerate(spy_hof['genes'][i]):
            row[f'gene_{j}'] = gene
        spy_data.append(row)
    
    df_spy = pd.DataFrame(spy_data)
    spy_csv_path = os.path.join(results_dir, "coevolution_spy_hof_no_archive.csv")
    df_spy.to_csv(spy_csv_path, index=False)
    print(f"Spy HoF exported to {spy_csv_path}")

if __name__ == "__main__":

    base_dir = os.path.join("results", "no_archive")
    os.makedirs(base_dir, exist_ok=True)

    evolutionary_runs = 10

    for run_idx in range(1, evolutionary_runs + 1):

        # store data in the run directory
        run_dir = os.path.join(base_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

         # Create instances of the EvolutionaryAlgorithm for both populations
        resistance_ea = EvolutionaryAlgorithm()
        spy_ea = EvolutionaryAlgorithm()

        print(f"Running evolutionary run {run_idx} of {evolutionary_runs}")
    
        # Initialize populations
        print("Initializing resistance population...")
        resistance_ea.initialize_population(
            population_size=21,  # 21 resistance fighters (mu)
            num_genes=14,        # 14 genes for resistance
            value_range=(-1, 1)  # Values between -1 and 1 for resistance
        )
    
        print("Initializing spy population...")
        spy_ea.initialize_population(
            population_size=21,  # 21 spies (mu)
            num_genes=10,        # 10 genes for spies
            value_range=(0, 1)   # Values between 0 and 1 for spies
        )
    
        # Print some information about the populations
        print(f"Number of resistance fighters: {len(resistance_ea.population)}")
        print(f"Number of spies: {len(spy_ea.population)}")
        print(f"Resistance gene size: {len(resistance_ea.population[0])}")
        print(f"Spy gene size: {len(spy_ea.population[0])}")
    
        # Co-evolve the populations
        print("\nCo-evolving populations...")
        final_resistance, final_spies, resistance_stats, spy_stats, resistance_hof, spy_hof = co_evolve(
            resistance_ea, spy_ea,
            num_gen=200, # number of generations
            mu=21,       # Population size
            lambda_=150, # Offspring size
            p_cross=0.1, # Crossover probability
            std=0.1,     # Standard deviation for mutation
            hof_size=100  # Hall of Fame size
        )

        # Export the statistics
        export_stats(resistance_stats, spy_stats, results_dir=run_dir)
    
        # Export the populations and Hall of Fame
        export_populations(final_resistance, final_spies, results_dir=run_dir)
    
        # Export HoFs
        export_hof(resistance_hof, spy_hof, results_dir=run_dir)
    
        # Visualize the co-evolution statistics
        viz_path = os.path.join(run_dir, f"coevolution_no_archive_run{run_idx}.png")
        visualize_coevolution(resistance_stats, spy_stats, save_path=viz_path) 

   