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

tasks_plot_color_map = {
    3: "red",
    4: "blue",
    5: "green"
}

def load_logbook_json(path):
    logbook_path = f"{path}/logbook.json"
    with open(logbook_path, "r") as f:
        logbook = json.load(f)
    return logbook

def load_experiment_json(path):
    experiment_path = f"{path}/experiment.json"
    with open(experiment_path, "r") as f:
        experiment = json.load(f)
    return experiment

def plot_drift(paths, tasks, name):
    """
    paths must be in the format:
    [[path1, path2, ...], [pathdrift1, pathdrift2, ...], [pathdriftdrift1, pathdriftdrift2, ...], ...]
    """
    
    assert len(paths) == len(tasks), "Paths and tasks must have the same length"

    counter_gens = 0
    baf = [] # best anytime fitness
    beef = [] # best end evolution fitness
    raf = [] # retaining anytime fitness
    reef = [] # retaining end evolution fitness

    plotted_task = []
    plotted_retaining = []
    best_instances = []
    best_retention_instances = []

    for i in range(len(paths)):

        # Iterate thru the drifts
        log_drift = []
        experiment_info_drift = []
        for j in range(len(paths[i])):
            # Iterates thru the experiments instances
            log = load_logbook_json(paths[i][j])
            log_drift.append(log)
            experiment_info = load_experiment_json(paths[i][j])
            experiment_info_drift.append(experiment_info)

        bests = []
        for log in log_drift:
            no_penalty = log.get("no_penalty", [])
            if no_penalty == []:
                bests.append(log["best"]) 
            else:
                bests.append(no_penalty)
        
        type_of_retaining = experiment_info_drift[0]["type_of_retaining"]
        best_instances.append(np.argmax([experiment_info["best_fitness"] for experiment_info in experiment_info_drift]))
        avg_bests = np.mean(bests, axis=0)

        current_gen = counter_gens
        counter_gens += len(avg_bests)
        current_color = tasks_plot_color_map[tasks[i]]

        baf.extend(avg_bests)
        beef.append(avg_bests[-1])
        
        # Plot best
        # check if the task was already plotted
        if current_color not in plotted_task:
            plt.plot(range(current_gen, counter_gens), avg_bests, color=current_color, label=f"Best ({current_color})")
            plotted_task.append(current_color)
        else:
            plt.plot(range(current_gen, counter_gens), avg_bests, color=current_color)

        for best in bests:
            plt.plot(range(current_gen, counter_gens), best, color=current_color, alpha=0.2)
        
        # Plot forgetting
        if i != 0:
            prev_color = tasks_plot_color_map[tasks[i-1]]
            
            
            forgetting = [log["retaining"] for log in log_drift]
            best_retention_instances.append(np.argmax([experiment_info["best_fitness_retaining_top"] for experiment_info in experiment_info_drift]))
            avg_forgetting = np.mean(forgetting, axis=0)
            # TODO: not needed if retaining is same of the plot, also can be retaing find bests
            retaining_end = [experiment_info["best_fitness_retaining_top"] for experiment_info in experiment_info_drift]
            avg_retaining_end = np.mean(retaining_end)
            
            raf.extend(avg_forgetting)
            reef.append(avg_retaining_end)
            
            # range from current to counter_gens but every 10 gens
            range_forgetting = np.arange(current_gen, counter_gens, 10)
            
            # Plot drifts lines
            if i == 1:
                plt.axvline(x=current_gen, linestyle = '--', color='grey', label=f"Drift")
            else:    
                plt.axvline(x=current_gen, linestyle = '--', color='grey')
            
            if prev_color not in plotted_retaining:
                plt.plot(range_forgetting, avg_forgetting, linestyle='-.', label=f"Retaining {type_of_retaining} ({prev_color})", color=prev_color)
                plotted_retaining.append(prev_color)
            else:
                plt.plot(range_forgetting, avg_forgetting, linestyle='-.', color=prev_color)
            
            for retaining in forgetting:
                plt.plot(range_forgetting, retaining, linestyle='-.', color=prev_color, alpha=0.2)
               
    baf = np.array(baf).flatten()
    print(f"Best Anytime Fitness: {np.mean(baf)}")
    print(f"Average End Evolution Fitness: {np.mean(beef)}")
    for i in range(len(best_instances)):
        print(f"Best seed for evo {i} at {best_instances[i]}") 
    
    raf = np.array(raf).flatten()
    print(f"Average Retaining End Evolution Fitness: {np.mean(reef)}")
    for i in range(len(best_retention_instances)):
        print(f"Best seed for retaining evo {i} at {best_retention_instances[i]}")

    plt.legend(loc="lower left", fontsize=8.5)
    if len(tasks) == 1:
        plt.title("Evolution")
    else:
        plt.title("Evolution with Drifts")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig(f"plots/{name}.png")
    plt.show()
    plt.clf()

    # Save the data in csv, read csv and write new record at the end of the csv
    # if not os.path.exists("drift.csv"):
    #     with open("drift.csv", "w") as f:
    #         f.write("Name, baf, avg_beef, raf, avg_reef, evo1_beef, evo2_beef, evo2_reef, evo3_beef, evo3_reef\n")
    
    # 
    # with open(f"results_drift{drift_string}.csv", "a") as f:
    #     row = f"{name}, {np.mean(baf)}, {np.mean(beef)}, {np.mean(raf)}, {np.mean(reef)}"
    #     row += f", {beef[0]}"
    #     for i in range(len(reef)):
    #         row += f", {beef[i+1]}, {reef[i]}"
    #     f.write(row + "\n")
    
    # TODO: rename
    drift_string = ""
    for task in tasks:
        if task != tasks[-1]:
            drift_string += "drift" + str(task) + "_"
        else:
            drift_string += "drift" + str(task)
    # do the same as the previous line but with pandas
    
    if not os.path.exists(f"results_{drift_string}.csv"):
        df = pd.DataFrame(columns=["Name", "baf", "avg_beef", "raf", "avg_reef", "evo1_beef", "evo2_beef", "evo2_reef", "evo3_beef", "evo3_reef"])
        df.to_csv(f"results_{drift_string}.csv", index=False)
    
    df = pd.read_csv(f"results_{drift_string}.csv")
    row = {"Name": name, "baf": np.mean(baf), "avg_beef": np.mean(beef), "raf": np.mean(raf), "avg_reef": np.mean(reef)}
    row["evo1_beef"] = beef[0]
    for i in range(len(reef)):
        row[f"evo{i+2}_beef"] = beef[i+1]
        row[f"evo{i+2}_reef"] = reef[i]
    new_df = pd.concat([df, pd.DataFrame.from_records([row])])
    new_df.to_csv(f"results_{drift_string}.csv", index=False)

if __name__ == "__main__":

    seeds = range(10)
    # Define the directory containing the files
    results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong evolutionary swarms/results")
    results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong evolutionary swarms/results/drift_search_coarse/results")

    tasks = [3, 4, 3]
    name = "cheat"
    # Regular expression pattern to match the desired part
    # pattern = r"^(?!.*drift).*$"

    # # Set to store unique parts of the filenames
    # unique_parts = set()

    # # Iterate through all files in the specified directory
    # for filename in os.listdir(results_path):
    #     # Check if the filename matches the pattern
    #     match = re.match(pattern, filename)
    #     if match:
    #         # If there's a match, add it to the set
    #         unique_parts.add(match.group()[:-2]) # remove the seed TODO: if seed is longer that 1 digit dont work

    # # Convert the set to a sorted list
    # exps = sorted(list(unique_parts))

    # # Print the result
    # print(exps)

    # exps = ["baselinefrgd3_neat_500_500_50_3_7_b", "baselinefrgd3_neat_500_500_50_3_7_u"]
    # path = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong\ evolutionary\ swarms/results" 
    # path = "results"
    # for exp in exps:
    #     name = exp
    #     drift1 = "drift34"
    #     drift2 = "drift43"
    #     tasks = [3, 4, 3]

    #     experiments_list = []
    #     drift_str = ""
    #     for i, task in enumerate(tasks):
    #         experiments = []
    #         if i != 0:
    #             drift_str = drift_str + f"_drift{tasks[i-1]}{tasks[i]}"
    #         for seed in seeds:
    #             if i == 0:
    #                 experiments.append(f"{path}/{name}_{seed}")
    #             else:
    #                 experiments.append(f"{path}/{name}_{seed}{drift_str}")
    #         experiments_list.append(experiments)
        
    #     plot_drift(experiments_list, tasks, name)
    experiments_list = [[
        f"{results_path}/gdcoarse5_neat_800_100_100_5_30_u_1",
        # f"{results_path}/bristol/bristol2gd3_neat_500_50_300_3_7_u_0",
        # f"{results_path}/bristol/bristol3gd3_neat_500_50_300_3_7_u_0"
        # f"{results_path}/bristol/gd_3seed3gd3_neat_500_50_300_3_7_u_3",
        # f"{results_path}/bristol/gd_3seed4gd3_neat_500_50_300_3_7_u_3",
        # f"{results_path}/bristol/gd_3seed5gd3_neat_500_50_300_3_7_u_3",
        # f"{results_path}/bristol/gd_3seed6gd3_neat_500_50_300_3_7_u_3",
        # f"{results_path}/bristol/gd_3seed7gd3_neat_500_50_300_3_7_u_3",
        # f"{results_path}/bristol/gd_3seed8gd3_neat_500_50_300_3_7_u_3",
        # f"{results_path}/bristol/gd_3seed9gd3_neat_500_50_300_3_7_u_3"
    ],
        [
        f"{results_path}/gdcoarse21_neat_800_100_100_5_30_u_1_drift34",
        # f"{results_path}/bristol/bristol2gd3_neat_500_50_300_3_7_u_0_drift34",
        # f"{results_path}/bristol/bristol3gd3_neat_500_50_300_3_7_u_0_drift34"
        # f"{results_path}/bristol/gd_3seed3gd3_neat_500_50_300_3_7_u_0_drift34",
        # f"{results_path}/bristol/gd_3seed4gd3_neat_500_50_300_3_7_u_0_drift34",
        # f"{results_path}/bristol/gd_3seed5gd3_neat_500_50_300_3_7_u_0_drift34",
        # f"{results_path}/bristol/gd_3seed6gd3_neat_500_50_300_3_7_u_0_drift34",
        # f"{results_path}/bristol/gd_3seed7gd3_neat_500_50_300_3_7_u_0_drift34",
        # f"{results_path}/bristol/gd_3seed8gd3_neat_500_50_300_3_7_u_0_drift34",
        # f"{results_path}/bristol/gd_3seed9gd3_neat_500_50_300_3_7_u_0_drift34"
    ],
    [
        f"{results_path}/gdcoarse5_neat_800_100_100_5_30_u_5_drift34_drift43",
        # f"{results_path}/bristol/bristol2gd3_neat_500_50_300_3_7_u_0_drift34_drift43",
        # f"{results_path}/bristol/bristol3gd3_neat_500_50_300_3_7_u_0_drift34_drift43"
        # f"{results_path}/bristol/gd_3seed3gd3_neat_500_50_300_3_7_u_0_drift34_drift43",
        # f"{results_path}/bristol/gd_3seed4gd3_neat_500_50_300_3_7_u_0_drift34_drift43",
        # f"{results_path}/bristol/gd_3seed5gd3_neat_500_50_300_3_7_u_0_drift34_drift43",
        # f"{results_path}/bristol/gd_3seed6gd3_neat_500_50_300_3_7_u_0_drift34_drift43",
        # f"{results_path}/bristol/gd_3seed7gd3_neat_500_50_300_3_7_u_0_drift34_drift43",
        # f"{results_path}/bristol/gd_3seed8gd3_neat_500_50_300_3_7_u_0_drift34_drift43",
        # f"{results_path}/bristol/gd_3seed9gd3_neat_500_50_300_3_7_u_0_drift34_drift43"
    ]      
    ]

    plot_drift(experiments_list, tasks, name)
                
