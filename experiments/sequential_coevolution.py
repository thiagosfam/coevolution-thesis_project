import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import matplotlib.ticker as ticker
from evolutionary.core import EvolutionaryAlgorithm

def co_evolve(resistance_ea, spy_ea, 
              num_gen=2000, mu=21, lambda_=150,
              p_cross=0.1, std=0.1):
    """
    Alternating co-evolution: evolve one population while the other is frozen,
    switching every 100 generations. Tracks both populations' fitness each generation.
    """

    resistance_stats = {'max_fitness': [], 'avg_fitness': [], 'std_fitness': []}
    spy_stats = {'max_fitness': [], 'avg_fitness': [], 'std_fitness': []}
    resistance_ancestral_line = {'generation': [], 'fitness': [], 'genes': []}
    spy_ancestral_line = {'generation': [], 'fitness': [], 'genes': []}

    resistance_pop = resistance_ea.population
    spy_pop = spy_ea.population
    spy_std = std

    for generation in range(num_gen):
        evolving_spies = (generation // 200) % 2 == 0
        print(f"\nGeneration {generation+1}/{num_gen} | Evolving: {'Spies' if evolving_spies else 'Resistance'}")

        if not evolving_spies:
            # === Evolve Resistance ===
            offspring = []
            for _ in range(lambda_ // 2):
                p1, p2 = resistance_ea.parent_selection(resistance_pop)
                c1, c2 = resistance_ea.uniform_crossover(p1, p2, p_cross)
                c1 = resistance_ea.gaussian_mutation(c1, 1.0, std, (-1, 1))
                c2 = resistance_ea.gaussian_mutation(c2, 1.0, std, (-1, 1))
                offspring += [c1, c2]

            evaluated = resistance_ea.evaluate_population(
                offspring, offspring, spy_pop, role='resistance'
            )
            survivors = resistance_ea.survival_selection(evaluated, mu)
            resistance_pop = [ind for ind, _ in survivors]

            fitnesses = [fit for _, fit in survivors]
            best_ind, best_fit = max(survivors, key=lambda x: x[1])
            resistance_stats['max_fitness'].append(max(fitnesses))
            resistance_stats['avg_fitness'].append(np.mean(fitnesses))
            resistance_stats['std_fitness'].append(np.std(fitnesses))
            resistance_ancestral_line['generation'].append(generation)
            resistance_ancestral_line['fitness'].append(best_fit)
            resistance_ancestral_line['genes'].append(best_ind)

            # Re-evaluate frozen spy population
            reeval = spy_ea.evaluate_population(spy_pop, spy_pop, resistance_pop, role='spy')
            fit_vals = [fit for _, fit in reeval]
            best, best_fit = max(reeval, key=lambda x: x[1])
            spy_stats['max_fitness'].append(max(fit_vals))
            spy_stats['avg_fitness'].append(np.mean(fit_vals))
            spy_stats['std_fitness'].append(np.std(fit_vals))
            spy_ancestral_line['generation'].append(generation)
            spy_ancestral_line['fitness'].append(best_fit)
            spy_ancestral_line['genes'].append(best)

        else:
            # === Evolve Spies ===
            offspring = []
            for _ in range(lambda_ // 2):
                p1, p2 = spy_ea.parent_selection(spy_pop)
                c1, c2 = spy_ea.uniform_crossover(p1, p2, p_cross)
                c1 = spy_ea.gaussian_mutation(c1, 1.0, spy_std, (0, 1))
                c2 = spy_ea.gaussian_mutation(c2, 1.0, spy_std, (0, 1))
                offspring += [c1, c2]

            evaluated = spy_ea.evaluate_population(
                offspring, offspring, resistance_pop, role='spy'
            )
            survivors = spy_ea.survival_selection(evaluated, mu)
            spy_pop = [ind for ind, _ in survivors]

            fitnesses = [fit for _, fit in survivors]
            best_ind, best_fit = max(survivors, key=lambda x: x[1])
            spy_stats['max_fitness'].append(max(fitnesses))
            spy_stats['avg_fitness'].append(np.mean(fitnesses))
            spy_stats['std_fitness'].append(np.std(fitnesses))
            spy_ancestral_line['generation'].append(generation)
            spy_ancestral_line['fitness'].append(best_fit)
            spy_ancestral_line['genes'].append(best_ind)

            spy_std *= 0.995  # decay mutation std

            # Re-evaluate frozen resistance population
            reeval = resistance_ea.evaluate_population(resistance_pop, resistance_pop, spy_pop, role='resistance')
            fit_vals = [fit for _, fit in reeval]
            best, best_fit = max(reeval, key=lambda x: x[1])
            resistance_stats['max_fitness'].append(max(fit_vals))
            resistance_stats['avg_fitness'].append(np.mean(fit_vals))
            resistance_stats['std_fitness'].append(np.std(fit_vals))
            resistance_ancestral_line['generation'].append(generation)
            resistance_ancestral_line['fitness'].append(best_fit)
            resistance_ancestral_line['genes'].append(best)

    return resistance_pop, spy_pop, resistance_stats, spy_stats, resistance_ancestral_line, spy_ancestral_line



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
    
    
    ax1.set_title('Co-Evolution of Resistance Fighters and Spies (Sequential)', fontsize=16)
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
    plt.show()

def export_stats(resistance_stats, spy_stats, results_dir="results"):
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
    }).to_csv(os.path.join(results_dir, "resistance_stats_sequential.csv"), index=False)
    print(f"Resistance statistics exported to {os.path.join(results_dir, 'resistance_stats_sequential.csv')}")
    
    # Export spy statistics
    print("\nExporting spy statistics to CSV...")
    pd.DataFrame({
        'max_fitness': spy_stats['max_fitness'],
        'avg_fitness': spy_stats['avg_fitness'],
        'std_fitness': spy_stats['std_fitness']
    }).to_csv(os.path.join(results_dir, "spy_stats_sequential.csv"), index=False)
    print(f"Spy statistics exported to {os.path.join(results_dir, 'spy_stats_sequential.csv')}")

def export_populations(resistance_pop, spy_pop, results_dir="results"):
    """
    Exports the final populations and Hall of Fame to CSV files.
    
    Args:
        resistance_pop: Final resistance population
        spy_pop: Final spy population
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
    resistance_csv_path = os.path.join(results_dir, "sequential_coevolution_final_resistance_population.csv")
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
    spy_csv_path = os.path.join(results_dir, "sequential_coevolution_final_spy_population.csv")
    df_spy.to_csv(spy_csv_path, index=False)
    print(f"Final spy population exported to {spy_csv_path}")  

def export_ancestral_lines(resistance_ancestral_line, spy_ancestral_line, results_dir="results"):
    """
    Exports the ancestral lines (best individual from each generation) to CSV files.
    
    Args:
        resistance_ancestral_line: Dictionary containing generation, fitness, and genes lists for resistance fighters
        spy_ancestral_line: Dictionary containing generation, fitness, and genes lists for spies
        results_dir: Directory to save the results
    """
    # Export the resistance ancestral line
    print("\nExporting resistance ancestral line to CSV...")
    resistance_data = []
    for i in range(len(resistance_ancestral_line['generation'])):
        row = {
            'generation': resistance_ancestral_line['generation'][i],
            'fitness': resistance_ancestral_line['fitness'][i]
        }
        # Add genes
        for j, gene in enumerate(resistance_ancestral_line['genes'][i]):
            row[f'gene_{j}'] = gene
        resistance_data.append(row)
    
    df_resistance = pd.DataFrame(resistance_data)
    resistance_csv_path = os.path.join(results_dir, "sequential_coevolution_resistance_ancestral_line.csv")
    df_resistance.to_csv(resistance_csv_path, index=False)
    print(f"Resistance ancestral line exported to {resistance_csv_path}")
    
    # Export the spy ancestral line
    print("Exporting spy ancestral line to CSV...")
    spy_data = []
    for i in range(len(spy_ancestral_line['generation'])):
        row = {
            'generation': spy_ancestral_line['generation'][i],
            'fitness': spy_ancestral_line['fitness'][i]
        }
        # Add genes
        for j, gene in enumerate(spy_ancestral_line['genes'][i]):
            row[f'gene_{j}'] = gene
        spy_data.append(row)
    
    df_spy = pd.DataFrame(spy_data)
    spy_csv_path = os.path.join(results_dir, "sequential_coevolution_spy_ancestral_line.csv")
    df_spy.to_csv(spy_csv_path, index=False)
    print(f"Spy ancestral line exported to {spy_csv_path}")

if __name__ == "__main__":

    base_dir = os.path.join("results", "sequential")
    os.makedirs(base_dir, exist_ok=True)

    evolutionary_runs = 10

    for run_idx in range(1, evolutionary_runs + 1):

        # Store data in appropriate directory
        run_dir = os.path.join(base_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

         # Create instances of the EvolutionaryAlgorithm for both populations
        resistance_ea = EvolutionaryAlgorithm()
        spy_ea = EvolutionaryAlgorithm()

        print(f"\n=== Starting Run {run_idx} of {evolutionary_runs}, saving into {run_dir} ===")
    
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
        final_resistance, final_spies, resistance_stats, spy_stats, resistance_ancestral_line, spy_ancestral_line = co_evolve(
            resistance_ea, spy_ea,
            num_gen=1200, # Number of generations
            mu=21,       # Population size
            lambda_=150, # Offspring size
            p_cross=0.1, # Crossover probability
            std=0.1,     # Standard deviation for mutation
        )
    
        # Export the statistics
        export_stats(resistance_stats, spy_stats, results_dir=run_dir)
    
        # Export the populations and Hall of Fame
        export_populations(final_resistance, final_spies, results_dir=run_dir)
    
        # Export ancestral lines
        export_ancestral_lines(resistance_ancestral_line, spy_ancestral_line, results_dir=run_dir)
    
        # Visualize the co-evolution statistics
       # viz_path = os.path.join(run_dir, f"coevolution_sequential_run{run_idx}.png")
       # visualize_coevolution(resistance_stats, spy_stats, save_path=viz_path)


   