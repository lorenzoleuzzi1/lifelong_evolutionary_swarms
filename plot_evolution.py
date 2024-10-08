# load log book json
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

tasks_plot_color_map = {
    3: "red",
    4: "blue",
    5: "green",
    6: "yellow",
    7: "purple"
}

def plot_evolutions_and_drifts(exp_paths, name):
    """
    drifts must be in the format:
    {
        [seed1_static, seed1_drift1, seed1_drift2, ...],
        [seed2_static, seed2_drift1, seed1_drift2, ...],
        [seed3_static, seed3_drift1, seed1_drift2, ...],
    }
    """
    # print(drifts)
    
    n_seeds = len(exp_paths)
    for d in exp_paths:
        if len(d) != len(exp_paths[0]):
            raise ValueError(f"All the experiment instances (seed) must have the \
                             same number of drifts. {len(d)} != {len(exp_paths[0])}")
    n_drifts = len(exp_paths[0])
    # TODO: check if they have the same number of generations
    # TODO: check if they have the same retention type

    baf_seed = [] # best anytime fitness per seed
    raf_seed = [] # retention anytime fitness per seed
    avg_beef_seed = [] # best end evolution fitness per seed
    avg_reef_seed = [] # retention end evolution fitness per seed
    beefs_seed = [] # best end evolution fitness per seed
    reefs_seed = [] # retention end evolution fitness per seed
    forgetting_seed = [] # forgetting per seed
    avg_forgetting_seed = [] # avg forgetting per seed
    best_whole_curves_perseed = []
    retention_whole_curves_perseed = []
    
    # Iterate thru the experiment instances (seeds)
    for i in range(n_seeds):
        
        best_whole_curve = []
        retention_whole_curve = []
        beefs = []
        reefs = []
        forgettings = []

        generations_counters = []
        generations_counters.append(0)
        task_colors = []
        
        # Iterate thru the drifts
        for j in range(n_drifts):
           
            log = load_logbook_json(exp_paths[i][j]) 
            info = load_experiment_json(exp_paths[i][j])

            # Check for incompatibilities between instances
            if i == 0:
                check_generations = info["generations"]
                check_retention = info["retention_type"]
                # TODO: maybe others 
            if check_generations != info["generations"] or \
                                (check_retention != info["retention_type"] and info["retention_type"] != None):
                print(i, check_generations, info["generations"], check_retention, info["retention_type"])
                raise ValueError(f"Experiments to plot must have the same number of generations and retention type")

            generations_counters.append(generations_counters[-1] + len(log["best"]))
            task_colors.append(tasks_plot_color_map[info["target_color"]])
             
            # Best metrics
            best_whole_curve.extend(log["best"])
            beefs.append(info["best_fitness"])

            # Plot best
            plt.plot(range(generations_counters[j], generations_counters[j+1]), 
                     log["best"], color=task_colors[j], alpha=0.2)

            # Retention metrics
            if info["retention_type"] is not None:
                retention_whole_curve.extend(log["retention"])
                retention_type = info["retention_type"]
                reefs.append(info[f"best_fitness_retention_{retention_type}"]) 

                forgettings.append(beefs[-2] - reefs[-1])

                # Plot retention TODO: maybe the last value is not in the plot because every n gens
                plt.plot(range(generations_counters[j], generations_counters[j+1], 10), 
                         log["retention"], linestyle='-.', color=task_colors[j-1], alpha=0.2)
        
        best_whole_curves_perseed.append(best_whole_curve) # For plotting the mean
        retention_whole_curves_perseed.append(retention_whole_curve) # For plotting the mean
        baf_seed.append(np.mean(best_whole_curve))
        raf_seed.append(np.mean(retention_whole_curve))
        avg_beef_seed.append(np.mean(beefs))
        avg_reef_seed.append(np.mean(reefs))
        beefs_seed.append(beefs)
        reefs_seed.append(reefs)
        forgetting_seed.append(forgettings)
        avg_forgetting_seed.append(np.mean(forgettings))
    
    # Plot the mean
    mean_best_whole_curves_perseed = np.mean(best_whole_curves_perseed, axis = 0)
    mean_retention_whole_curves_perseed = np.mean(retention_whole_curves_perseed, axis = 0)

    # Save the mean in a json
    
    plotted_best_tasks = []
    plotted_retention_tasks = []
    
    for i in range(n_drifts):

        drift_best_generations_range = range(generations_counters[i], generations_counters[i+1])
        mean_best_to_plot = mean_best_whole_curves_perseed[drift_best_generations_range]

        if i >= 1:
            if i == 1:
                plt.axvline(x=generations_counters[i], linestyle = '--', color='grey', label=f"Drift") 
            else:
                plt.axvline(x=generations_counters[i], linestyle = '--', color='grey')

        if task_colors[i] not in plotted_best_tasks:
            plt.plot(drift_best_generations_range, 
                     mean_best_to_plot, color=task_colors[i],
                     label=f"Best ({task_colors[i]})")
            plotted_best_tasks.append(task_colors[i])
        else:
            plt.plot(drift_best_generations_range, 
                        mean_best_to_plot, color=task_colors[i])
        
        if i != 0 and info["retention_type"] is not None:
            drift_retention_generations_range = range(generations_counters[i], generations_counters[i+1], 10)
            start_gen_retention = int(generations_counters[i-1] / 10) # 10 is the frequency of eval
            end_gen_retention = int(generations_counters[i] / 10)
            mean_retention_to_plot = mean_retention_whole_curves_perseed[start_gen_retention:end_gen_retention]
            
            if task_colors[i-1] not in plotted_retention_tasks:
                plt.plot(drift_retention_generations_range,
                         mean_retention_to_plot, linestyle='-.', color=task_colors[i-1],
                         label=f"Retention {retention_type} ({task_colors[i-1]})")
                plotted_retention_tasks.append(task_colors[i-1])
            else:
                plt.plot(drift_retention_generations_range, 
                         mean_retention_to_plot, linestyle='-.', color=task_colors[i-1])

    plt.legend(loc="lower left", fontsize=8.5)
    if n_drifts == 1:
        plt.title("Evolution")
    else:
        plt.title("Evolution with Drifts")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig(f"{name}/experiment_plot.png")
    # plt.show()
    plt.clf()

    # Save the metrics in a csv
    # Every row is a seed and the columns are the metrics
    columns = ["Name", "baf", "avg_beef", "raf", "avg_reef", "avg_forgetting"]
    for i in range(n_drifts): # Add the beef and reef for each drift
        columns.append(f"evo{i+1}_beef")
        if i > 0:
            columns.append(f"evo{i+1}_reef")
            columns.append(f"evo{i+1}_forgetting")
    
    data = []
    for i in range(n_seeds):
        row = [f"seed{i}"]
        row.append(baf_seed[i])
        row.append(avg_beef_seed[i])
        row.append(raf_seed[i])
        row.append(avg_reef_seed[i])
        row.append(avg_forgetting_seed[i])
        for j in range(n_drifts):
            row.append(beefs_seed[i][j])
            if j > 0:
                if reefs_seed[i] != []:
                        row.append(reefs_seed[i][j-1])
                else:
                    row.append(np.nan)
                if forgetting_seed[i] != []:
                    
                        row.append(forgetting_seed[i][j-1])
                else:
                    row.append(np.nan)
        data.append(row)

    # Add a mean row
    row = ["mean"]
    row.append(np.mean(baf_seed))
    row.append(np.mean(avg_beef_seed))
    row.append(np.mean(raf_seed))
    row.append(np.mean(avg_reef_seed))
    row.append(np.mean(avg_forgetting_seed))
    for j in range(n_drifts):
        row.append(np.mean([beefs[j] for beefs in beefs_seed]))
        if j > 0:
            if reefs_seed[0] != []:
                row.append(np.mean([reefs[j-1] for reefs in reefs_seed]))
            else:
                row.append(np.nan)
            if forgetting_seed[0] != []:
                row.append(np.mean([forgetting[j-1] for forgetting in forgetting_seed]))
            else:
                row.append(np.nan)
    data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{name}/experiment_metrics.csv", index=False)
    

if __name__ == "__main__":

    # seeds = range(10)
    # Define the directory containing the files
    results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results/results")
    # results_path = "results"
    experiments_name = "retention_pop_g_2" # TODO as required input argparse

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

        plot_evolutions_and_drifts(experiment_paths, f"{results_path}/{experiments_name}/{exp}")