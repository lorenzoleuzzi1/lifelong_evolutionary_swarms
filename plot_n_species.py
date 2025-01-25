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
def plot_n_species(experiment_paths_classic, experiment_paths_reg, save_path):
    """
    drifts must be in the format:
    {
        [seed1_static, seed1_drift1, seed1_drift2, ...],
        [seed2_static, seed2_drift1, seed1_drift2, ...],
        [seed3_static, seed3_drift1, seed1_drift2, ...],
    }
    """
    if len(experiment_paths_classic) != len(experiment_paths_reg):
        raise ValueError(f"All the experiment must have the \
                         same number of seeds. {len(experiment_paths_classic)} != {len(experiment_paths_reg)}")
    n_seeds = len(experiment_paths_classic)
    n_drifts = min(len(experiment_paths_classic[0]), len(experiment_paths_reg[0]))
    
    n_species_c_perseed = []
    n_species_r_perseed = []

    r_counter = range(200, 400)
   
    # Plot number of species at each generations
    for i in range(n_seeds):
        n_species_c = []
        n_species_r = []
        for j in range(n_drifts):
            
            with open(experiment_paths_classic[i][j] + "/logbook_species.json", "r") as f:
                log_species_c = json.load(f)

            with open(experiment_paths_reg[i][j] + "/logbook_species.json", "r") as f:
                log_species_r = json.load(f)
            
            for gen in log_species_c:
                n_species_c.append(len(gen))
            
            for gen in log_species_r:
                n_species_r.append(len(gen))
        
        plt.plot(n_species_c, alpha=0.1, color='#11978D') # 
        plt.plot(r_counter, n_species_r[200:400], alpha=0.1, color='#D95319')
        n_species_c_perseed.append(n_species_c)
        n_species_r_perseed.append(n_species_r)
    
    mean_n_species_c = np.mean(n_species_c_perseed, axis=0)
    mean_n_species_r = np.mean(n_species_r_perseed, axis=0)
    plt.axvline(x=200, linestyle = '--', color='grey')
    # plt.axvline(x=400, linestyle = '--', color='grey')
    plt.plot(mean_n_species_c, color='#11978D', label="Classic")
    
    plt.plot(r_counter, mean_n_species_r[200:400], color='#D95319', label="Regularized")
    plt.xlabel('Generations', fontsize=14)
    plt.ylabel("Number of Species", fontsize=14)
    plt.ylim(0, 10)
    plt.yticks(range(0, 11))
    plt.legend(fontsize=12)
    # Save
    plt.savefig(f"{save_path}/n_species.png")
    plt.close()


if __name__ == "__main__":
    # Define the directory containing the files
    results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results_gecco4")
    
    c = f"{results_path}/snz_baselines"
    r = f"{results_path}/reg_gd_snz_11"
    # results_path = "results"
    print(f"Plotting species n_species from \n {c} and \n {r}")

    # Read all the files in the directory
    experiments_directories_c = [
        item for item in os.listdir(c) 
        if os.path.isdir(os.path.join(c, item)) 
        and not item.startswith('.')
    ]
    experiments_directories_r = [
        item for item in os.listdir(r) 
        if os.path.isdir(os.path.join(r, item)) 
        and not item.startswith('.')
    ]

    
    for exp_c, exp_r in zip(experiments_directories_c, experiments_directories_r):
        exp_seeds_directories_c = [
            item for item in os.listdir(f"{c}/{exp_c}") 
            if os.path.isdir(os.path.join(f"{c}/{exp_c}", item)) 
            and not item.startswith('.')
        ]        
        exp_seeds_directories_c = sorted(exp_seeds_directories_c)

        exp_seeds_directories_r = [
            item for item in os.listdir(f"{r}/{exp_r}")
            if os.path.isdir(os.path.join(f"{r}/{exp_r}", item))
            and not item.startswith('.')
        ]
        exp_seeds_directories_r = sorted(exp_seeds_directories_r)

        experiment_paths_c = []
        experiment_paths_r = []
        
        for exp_seed_c, exp_seed_r in zip(exp_seeds_directories_c, exp_seeds_directories_r):
            print("\t", exp_seed_c, exp_seed_r)
            path_c = f"{c}/{exp_c}/{exp_seed_c}"
            exp_seed_drifts_c = [
                item for item in os.listdir(path_c) 
                if os.path.isdir(os.path.join(path_c, item)) and not item.startswith('.')
            ]   
            path_r = f"{r}/{exp_r}/{exp_seed_r}"
            exp_seed_drifts_r = [
                item for item in os.listdir(path_r)
                if os.path.isdir(os.path.join(path_r, item)) and not item.startswith('.')
            ]
            exp_seed_drifts_c = sorted(exp_seed_drifts_c, key=lambda e: len(e))
            experiment_paths_c.append([f"{path_c}/{e}" for e in exp_seed_drifts_c])
            exp_seed_drifts_r = sorted(exp_seed_drifts_r, key=lambda e: len(e))
            experiment_paths_r.append([f"{path_r}/{e}" for e in exp_seed_drifts_r])
            
        plot_n_species(experiment_paths_c, experiment_paths_r, f"{results_path}")

        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "size")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "retention")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "adjusted_fitness")
        # plot_species(experiment_paths, f"{results_path}/{experiments_name}/{exp}", "fitness")