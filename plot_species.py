import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from deap import base
import pandas as pd
import os
import re
from utils import load_logbook_json, load_experiment_json
# TODO!!!!!
y_index_map = {
    "fitness": 0,
    "retention": 1
}

# THIS THING CAN BE DONE FOR A SINGLE RUN
def plot_species(experiment_paths, experiment_name, y):
    """
    drifts must be in the format:
    {
        [seed1_static, seed1_drift1, seed1_drift2, ...],
        [seed2_static, seed2_drift1, seed1_drift2, ...],
        [seed3_static, seed3_drift1, seed1_drift2, ...],
    }
    """
    if y != "size" and y != "n_species":
        y_index = y_index_map[y]
    # print(drifts)
    y_index = 1
    n_seeds = len(experiment_paths)
    for d in experiment_paths:
        if len(d) != len(experiment_paths[0]):
            raise ValueError(f"All the experiment instances (seed) must have the \
                             same number of drifts. {len(d)} != {len(experiment_paths[0])}")
    n_drifts = len(experiment_paths[0])


    if y == "n_species":
        n_species_perseed = []
        # Plot number of species at each generations
        for i in range(n_seeds):
            n_species = []
            for j in range(n_drifts):
                
                with open(experiment_paths[i][j] + "/logbook_species.json", "r") as f:
                    log_species = json.load(f)
                
                for gen in log_species:
                    n_species.append(len(gen))
            plt.plot(n_species, alpha=0.1, color='#D95319')
            n_species_perseed.append(n_species)
        
        mean_n_species = np.mean(n_species_perseed, axis=0)
        plt.axvline(x=200, linestyle = '--', color='grey')
        plt.axvline(x=400, linestyle = '--', color='grey')
        plt.plot(mean_n_species, color='#D95319')
        plt.xlabel('Generations')
        plt.ylabel("Number of Species")
        plt.ylim(0, 10)
        plt.yticks(range(0, 11))
        # Save
        plt.savefig(f"{experiment_name}/{y}.png")
        plt.close()

        return

    # Iterate thru the experiment instances (seeds)
    for i in range(n_seeds):

        generations_counters = []
        generations_counters.append(0)
        
        generations_counter = 0
        log_to_plot = {}
        generations_to_plot = {}
        log_n_species_to_plot = []
        # Iterate thru the drifts
        for j in range(n_drifts):
            if j != 0:
                # plot vertical line
                plt.axvline(x=generations_counter, linestyle = '--', color='grey')
            
            with open(experiment_paths[i][j] + "/logbook_generations.json", "r") as f:
                log_generations = json.load(f)
            with open(experiment_paths[i][j] + "/logbook_species.json", "r") as f:
                log_species = json.load(f)

            log_full = []
            # Add species to logbook generations
            for gen, gen_species in zip(log_generations, log_species):
                dict_gen = {}
                for species in gen_species:
                    dict_gen[species] = {}
                    for genome_in_that_species in gen_species[species]:
                        dict_gen[species][genome_in_that_species] = gen[genome_in_that_species]
                log_full.append(dict_gen)
            
            # Lets plot the species
            for gen in log_full:
                for species in gen:
                    log_to_plot[species] = log_to_plot.get(species, [])
                    generations_to_plot[species] = generations_to_plot.get(species, [])

                    if y == "size":
                        log_to_plot[species].append(len(gen[species]))
                        generations_to_plot[species].append(generations_counter)
                        continue

                    # do the average of y_index
                    # Check if the len of the list has the y index
                    if len(gen[species][list(gen[species].keys())[0]]) <= y_index:
                        continue
                    y_values = [gen[species][genome][y_index] for genome in gen[species]]
                    y_mean = np.mean(y_values)
                    log_to_plot[species].append(y_mean)
                    generations_to_plot[species].append(generations_counter)
                generations_counter += 1

            generations_counters.append(generations_counter)
        
        # Select best 5 species
        best_species = {}

        for species in log_to_plot:
            if log_to_plot[species] == []:
                continue
            best_species[species] = log_to_plot[species][-1]

        best_species = dict(sorted(best_species.items(), key=lambda item: item[1], reverse=True)[:5])

        # Select worst 5 species
        worst_species = {}

        for species in log_to_plot:
            if log_to_plot[species] == []:
                continue
            worst_species[species] = log_to_plot[species][-1]

        worst_species = dict(sorted(worst_species.items(), key=lambda item: item[1], reverse=False)[:5])

        # Plot the best species
        for species in best_species:
            plt.plot(generations_to_plot[species], log_to_plot[species], label=species)

        # plt.xlabel('Generations')
        # plt.ylabel(y)
        # plt.title(f"Best Species {y}")
        # plt.legend()
        # plt.show()
        # plt.close()

        # Plot the worst species
        for species in worst_species:
            plt.plot(generations_to_plot[species], log_to_plot[species], label=species)

        plt.xlabel('Generations')
        plt.ylabel(y)
        plt.title(f"Species {y}")
        plt.legend()
        plt.show()
        plt.close()

        # Plot the species
        for species in log_to_plot:
            plt.plot(generations_to_plot[species], log_to_plot[species], label=species)
        
        plt.xlabel('Generations')
        plt.ylabel(y)
        plt.title(f"Species {y}")
        plt.legend()
        plt.show()
        plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the evolution of the experiments.')
    parser.add_argument('experiment_name', type=str, help=f'The name of the experiments.')
    parser.add_argument('y', type=str, help=f'The name of the y axis. Must be in fitness or size.')
    # seeds = range(10)
    # Define the directory containing the files
    results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results_gecco4")
    # results_path = "results"
    experiments_name = parser.parse_args().experiment_name
    y = parser.parse_args().y
    if y not in ["fitness", "size", "retention", "adjusted_fitness", "n_species"]:
        raise ValueError("y must be in fitness or size")
    
    print(f"Plotting species {y} for {experiments_name}")

    # Read all the files in the directory
    experiments_directories = [
        item for item in os.listdir(f"{results_path}/{experiments_name}") 
        if os.path.isdir(os.path.join(f"{results_path}/{experiments_name}", item)) 
        and not item.startswith('.')
    ]
    
    for exp in experiments_directories:
        exp_seeds_directories = [
            item for item in os.listdir(f"{results_path}/{experiments_name}/{exp}") 
            if os.path.isdir(os.path.join(f"{results_path}/{experiments_name}/{exp}", item)) 
            and not item.startswith('.')
        ]        
        exp_seeds_directories = sorted(exp_seeds_directories)
        print(exp)
        experiment_paths = []
        
        for exp_seed in exp_seeds_directories:
            print("\t", exp_seed)
            path = f"{results_path}/{experiments_name}/{exp}/{exp_seed}"
            exp_seed_drifts = [
                item for item in os.listdir(path) 
                if os.path.isdir(os.path.join(path, item)) and not item.startswith('.')
            ]            
            exp_seed_drifts = sorted(exp_seed_drifts, key=lambda e: len(e))
            experiment_paths.append([f"{path}/{e}" for e in exp_seed_drifts])

        plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", y)
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "size")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "retention")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "adjusted_fitness")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "fitness")