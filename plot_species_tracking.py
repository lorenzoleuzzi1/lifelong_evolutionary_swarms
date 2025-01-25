import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from deap import base
import pandas as pd
import os
from utils import load_logbook_json, load_experiment_json
# TODO!!!!!
y_index_map = {
    "fitness": 0,
    "retention": 1
}

# THIS THING CAN BE DONE FOR A SINGLE RUN
def plot_tracking(drifts_directories):
  
    species = {}
    print(drifts_directories)
    gen_counter = 0
    for j in range(len(drifts_directories)):
            
        with open(drifts_directories[j] + "/logbook_species.json", "r") as f:
            log_species = json.load(f)

        for k, gen in enumerate(log_species):
            for species_id in gen:
                
                if species_id not in species:
                    species[species_id] = []
                
                species[species_id].append(k + gen_counter)
        
        gen_counter += len(log_species)
    
    # Plot species tracking
    # x is the generation
    # and y is the species id so its like a binary matrix
    for species_id in species:
        y = [species_id] * len(species[species_id])
        plt.plot(species[species_id], y, color='#11978D')
    
    plt.axvline(x=200, linestyle = '--', color='grey')
    plt.axvline(x=400, linestyle = '--', color='grey')

    plt.xlabel('Generations')
    plt.ylabel("Species ID")
    plt.yticks([0, len(species)-1])
    # Resize to be shorter and longer not squared
    plt.gcf().set_size_inches(6, 7)
    plt.savefig(f"{drifts_directories[0]}/species_tracking.png")


if __name__ == "__main__":
    # Define the directory containing the files
    results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results_gecco4")
    
    # experiment_run = f"{results_path}/reg_gd_snz_11/11/seed13"
    experiment_run = f"{results_path}/snz_baselines/neat_500_200_300_5_20_10/seed13"

    print(f"Plotting species tracking from \n {experiment_run}")


    drifts_directories = [
        item for item in os.listdir(experiment_run) 
        if os.path.isdir(os.path.join(experiment_run, item)) 
        and not item.startswith('.')
    ]
    # sort
    drifts_directories = sorted(drifts_directories)
    print(drifts_directories)
    # add the results path
    drifts_directories = [f"{experiment_run}/{d}" for d in drifts_directories]

    plot_tracking(drifts_directories)

        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "size")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "retention")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "adjusted_fitness")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "fitness")