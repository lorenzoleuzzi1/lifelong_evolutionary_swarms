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
N_PREVS = 1
NO_PENALTY = "best_no_penalty" # "best_no_penalty" "no_penalty"
PREV_TARGETS = "prev_target_colors" # "prev_target_colors" "prev_target_color"
task_color_map = {
    3: "red",
    4: "blue",
    5: "green",
    6: "yellow",
    7: "purple",
    8: "white",
    9: "#0072BD", # cyan
    10: "black"
}

color_task_map = {
    "red": 3,
    "blue": 4,
    "green": 5,
    "yellow": 6,
    "purple": 7,
    "white": 8,
    "#0072BD": 9, # cyan
    "black": 10
}

map_title_str = {
    "ga": "GA",
    "evostick": "EvoStick",
    "cma-es": "CMA-ES",
    "neat": "NEAT",
    "u": "Uniform",
    "b": "Biased",
}

def plot_evolutions_and_drifts(exp_paths, name, retention_type):
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
    seeds = [-1 for i in range(n_seeds)]
    # TODO: check if they have the same number of generations

    baf_seed = [] # best anytime fitness per seed
    avg_beef_seed = [] # best end evolution fitness per seed
    beefs_seed = [] # best end evolution fitness per seed
    reefs_seed = {
        3 : [],
        4 : [],
        5 : [],
        6 : [],
        7 : []
    } # retention end evolution fitness per seed
    forgetting_seed = {
        3 : [],
        4 : [],
        5 : [],
        6 : [],
        7 : []
    } # forgetting per seed

    best_whole_curves_perseed = []
    retention_whole_curves_perseed = {
        3 : [],
        4 : [],
        5 : [],
        6 : [],
        7 : []
    }

    
    # --- PLOT ---
    # Iterate thru the experiment instances (seeds)
    for i in range(n_seeds):
        
        best_whole_curve = []
        retention_whole_curve = {
            3 : [],
            4 : [],
            5 : [],
            6 : [],
            7 : []
        }
        beefs = []
        reefs = {
            3 : [],
            4 : [],
            5 : [],
            6 : [],
            7 : []
        }
        forgettings = {
            3 : [],
            4 : [],
            5 : [],
            6 : [],
            7 : []
        }

        generations_counters = []
        generations_counters.append(0)
        task_colors = []
        
        # Iterate thru the drifts
        for j in range(n_drifts):
           
            with open(experiment_paths[i][j] + "/logbook_summary.json", "r") as f:
                log = json.load(f)
            with open(experiment_paths[i][j] + "/info.json", "r") as f:
                info = json.load(f)
            seeds[i] = info["seed"]

            # Check for incompatibilities between instances
            if i == 0:
                check_generations = info["generations"]
            if check_generations != info["generations"]:
                raise ValueError(f"Experiments to plot must have the same number of generations and retention type, got {check_generations} and {info['generations']}")

            generations_counters.append(generations_counters[-1] + len(log["best"]))
            task_colors.append(task_color_map[info["target_color"]])
             
            # Best metrics
            # if True:
            if log[NO_PENALTY] == []:
                best_whole_curve.extend(log["best"])
                beefs.append(info["best"])
                # Plot best
                plt.plot(range(generations_counters[j], generations_counters[j+1]), 
                    log["best"], color=task_colors[j], alpha=0.1)
            else:
                best_whole_curve.extend(log[NO_PENALTY])
                beefs.append(log[NO_PENALTY][-1])
                # Plot best
                plt.plot(range(generations_counters[j], generations_counters[j+1]),
                    log[NO_PENALTY], color=task_colors[j], alpha=0.1)
            
            if j > 0:
                
                if retention_type is not None:
                    for prev_target in info[PREV_TARGETS][-N_PREVS:]:
                        if log[f"retention_{retention_type}_{prev_target}"] == []:
                            print(j, retention_type, log[f"retention_{retention_type}"])
                            raise ValueError(f"Retention type {retention_type} not found in the logbook.")
                    
                        # Retention metrics
                        retention_whole_curve[prev_target].extend(log[f"retention_{retention_type}_{prev_target}"])
                        reefs[prev_target].append(info[f"retention_{retention_type}_{prev_target}"])

                        forgettings[prev_target].append(beefs[-2] - reefs[prev_target][-1])

                        plt.plot(range(generations_counters[j], generations_counters[j+1] + 10, 10), # TODO: change 10 with the frequency of eval
                                log[f"retention_{retention_type}_{prev_target}"], linestyle='-.', color=task_color_map[prev_target], alpha=0.1)
                
        
        best_whole_curves_perseed.append(best_whole_curve) # For plotting the mean
        baf_seed.append(np.mean(best_whole_curve))
        avg_beef_seed.append(np.mean(beefs))
        beefs_seed.append(beefs)

        for prev_target in info[PREV_TARGETS]:
            retention_whole_curves_perseed[prev_target].append(retention_whole_curve[prev_target]) # For plotting the mean
            reefs_seed[prev_target].append(reefs[prev_target])
            forgetting_seed[prev_target].append(forgettings[prev_target])

    
    # Plot the mean
    mean_best_whole_curves_perseed = np.mean(best_whole_curves_perseed, axis = 0)
    mean_retention_whole_curves_perseed = {
        3 : [],
        4 : [],
        5 : [],
        6 : [],
        7 : []
    }
    
    for prev_target in info[PREV_TARGETS]:
        mean_retention_whole_curves_perseed[prev_target] = np.mean(retention_whole_curves_perseed[prev_target], axis = 0)
    
    # mean_retention_whole_curves_perseed = np.mean(retention_whole_curves_perseed, axis = 0)
    
    start_gen_retention = {
        3 : 0,
        4 : 0,
        5 : 0,
        6 : 0,
        7 : 0
    }

    retention_task_color = []

    for i in range(n_drifts):

        drift_best_generations_range = range(generations_counters[i], generations_counters[i+1])
        mean_best_to_plot = mean_best_whole_curves_perseed[drift_best_generations_range]
        
        if i > 0:
            plt.axvline(x=generations_counters[i], linestyle = '--', color='grey')

        plt.plot(drift_best_generations_range, mean_best_to_plot, color=task_colors[i])
        
        if i != 0 and retention_type is not None:
            for prev_target in retention_task_color[-N_PREVS:]:
                len_retentions = 21 # int(len(mean_retention_whole_curves_perseed[prev_target]) / (n_drifts-1)) # TODO: 10 frequency param

                drift_retention_generations_range = range(generations_counters[i], generations_counters[i+1]+10, 10)
                start = start_gen_retention[prev_target]
                end = start_gen_retention[prev_target] + len_retentions
                mean_retention_to_plot = mean_retention_whole_curves_perseed[prev_target][start:end]
                start_gen_retention[prev_target] += len_retentions
                plt.plot(drift_retention_generations_range, 
                            mean_retention_to_plot, linestyle='-.', color=task_color_map[prev_target])
        
        retention_task_color.append(color_task_map[task_colors[i]])
    
    
    if n_drifts > 1 and retention_type is not None:
        # plt.legend(loc="lower left", fontsize=8.5)
        legend_elements = [plt.Line2D([0], [0], color='grey', linestyle='-', label='Current'),
                        plt.Line2D([0], [0], color='grey', linestyle='-.', label='Retention')]

        plt.legend(handles=legend_elements, loc='upper left', fontsize=12)

    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)

    # set y axis
    plt.ylim(0, 40)
    
    if retention_type is not None:
        plt.savefig(f"{name}/experiment_plot_retention_{retention_type}.png", bbox_inches='tight')
    else:
        plt.savefig(f"{name}/experiment_plot.png", bbox_inches='tight')
    # plt.show()
    plt.clf()
    # ------------

    # --- METRICS IN CSV ---
    # Save the metrics in a csv
    # Every row is a seed and the columns are the metrics
    columns = ["Name", "baf", "avg_bef", "avg_ref", "avg_f"] 
    for prev_target in info[PREV_TARGETS]:
        columns.append(f"avg_ref_{task_color_map[prev_target]}")
        columns.append(f"avg_f_{task_color_map[prev_target]}")
    
    csv_retention = []
    for i in range(n_drifts): # Add the beef and reef for each drift
        
        columns.append(f"evo{i+1}_bef")
        if i > 0:
            for prev_target in csv_retention[-N_PREVS:]:
                columns.append(f"evo{i+1}_ref_{task_color_map[prev_target]}")
                columns.append(f"evo{i+1}_f_{task_color_map[prev_target]}")
        
        csv_retention.append(color_task_map[task_colors[i]])
    
    data = []
    
    for i in range(n_seeds):
        row = [f"{seeds[i]}"]
        row.append(baf_seed[i])
        row.append(avg_beef_seed[i])

        avg_ref = 0
        avg_f = 0
        for prev_target in info[PREV_TARGETS]:
            avg_ref += np.mean(reefs_seed[prev_target][i])
            avg_f += np.mean(forgetting_seed[prev_target][i])
        row.append(avg_ref / len(info[PREV_TARGETS]))
        row.append(avg_f / len(info[PREV_TARGETS]))
        
        csv_prev_targets = []
        counter_colors = {
            3 : 0,
            4 : 0, 
            5 : 0,
            6 : 0,
            7 : 0
        }

        for prev_targets in info[PREV_TARGETS]:
            row.append(np.mean(reefs_seed[prev_targets][i]))
            row.append(np.mean(forgetting_seed[prev_targets][i]))

        for j in range(n_drifts):
            row.append(beefs_seed[i][j])
            if j > 0:
                for prev_target in csv_prev_targets[-N_PREVS:]:
                    if reefs_seed[prev_target][i] != []:
                        row.append(reefs_seed[prev_target][i][counter_colors[prev_target]])
                        row.append(forgetting_seed[prev_target][i][counter_colors[prev_target]])
                        counter_colors[prev_target] += 1
                    else:
                        row.append(np.nan)
                        row.append(np.nan)


            csv_prev_targets.append(color_task_map[task_colors[j]])
            
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    # Add row with the mean
    mean_row = df.mean()
    mean_row["Name"] = "mean"
    df.loc['mean'] = mean_row

    if retention_type is not None:
        df.to_csv(f"{name}/experiment_metrics_retention_{retention_type}.csv", index=False)
    else:
        df.to_csv(f"{name}/experiment_metrics.csv", index=False)
    
    # ----------------------
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Plot the evolution of the experiments.')
    parser.add_argument('experiment_name', type=str, help=f'The name of the experiments.')
    parser.add_argument('--retention_type', '-ret', type=str, default=None, help=f'The type of retention to plot.')
    # seeds = range(10)
    # Define the directory containing the files
    results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results_gecco4")
    # results_path = "results"
    experiments_name = parser.parse_args().experiment_name
    retention_type = parser.parse_args().retention_type

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

        plot_evolutions_and_drifts(experiment_paths, f"{results_path}/{experiments_name}/{exp}", retention_type)