# load log book json
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from deap import base
import pandas as pd

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

    for i in range(len(paths)):

        # Iterate thru the drifts
        log_drift = []
        for j in range(len(paths[i])):
            # Iterates thru the experiments instances
            log = load_logbook_json(paths[i][j])
            log_drift.append(log)

        bests = []
        for log in log_drift:
            no_penalty = log.get("no_penalty", [])
            if no_penalty == []:
                bests.append(log["best"]) 
            else:
                bests.append(no_penalty)
        
        type_of_retaining = log_drift[0]["type_of_retaining"]
        avg_bests = np.mean(bests, axis=0)

        current_gen = counter_gens
        counter_gens += len(avg_bests)
        current_color = tasks_plot_color_map[tasks[i]]

        baf.extend(avg_bests) # TODO: other measures
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
            avg_forgetting = np.mean(forgetting, axis=0)
            
            raf.extend(avg_forgetting)
            reef.append(avg_forgetting[-1])
            
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
    raf = np.array(raf).flatten()
    print(f"Retaining Anytime Fitness: {np.mean(raf)}")
    print(f"Average Retaining End Evolution Fitness: {np.mean(reef)}")    
    plt.legend(loc="lower left", fontsize=8.5)
    if len(tasks) == 1:
        plt.title("Evolution")
    else:
        plt.title("Evolution with Drifts")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig(f"{name}.png")
    plt.show()

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
    

    drift_string = ""
    for task in tasks:
        if task != tasks[-1]:
            drift_string += "drift" + str(task) + "_"
        else:
            drift_string += "drift" + str(task)
    # do the same as the previous line but with pandas
    
    if not os.path.exists(f"results_{drift_string}.csv"):
        df = pd.DataFrame(columns=["Name", "baf", "avg_beef", "raf", "avg_reef", "evo1_beef", "evo2_beef", "evo2_reef", "evo3_beef", "evo3_reef"])
        df.to_csv(f"results_drift{drift_string}.csv", index=False)
    
    df = pd.read_csv(f"results_drift{drift_string}.csv")
    row = {"Name": name, "baf": np.mean(baf), "avg_beef": np.mean(beef), "raf": np.mean(raf), "avg_reef": np.mean(reef)}
    row["evo1_beef"] = beef[0]
    for i in range(len(reef)):
        row[f"evo{i+2}_beef"] = beef[i+1]
        row[f"evo{i+2}_reef"] = reef[i]
    df = pd.concat([df, pd.DataFrame.from_records([row])])
    df.to_csv(f"results_drift{drift_string}.csv", index=False)

if __name__ == "__main__":
    # path = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong\ evolutionary\ swarms/results" 
    path = "results"
    name = "newgdregbaseline_neat_800_100_300_5_30_u"
    drift1 = "drift34"
    drift2 = "drift43"
    tasks = [3, 4, 3]
    experiments_list = [[f"{path}/{name}_1",
                f"{path}/{name}_2",
                f"{path}/{name}_3",
                f"{path}/{name}_4",
                f"{path}/{name}_5"
                ]
                ,
                [f"{path}/{name}_1_{drift1}",
                f"{path}/{name}_2_{drift1}",
                f"{path}/{name}_3_{drift1}",
                f"{path}/{name}_4_{drift1}",
                f"{path}/{name}_5_{drift1}"
                ]
                ,
                [f"{path}/{name}_1_{drift1}_{drift2}",
                f"{path}/{name}_2_{drift1}_{drift2}",
                f"{path}/{name}_3_{drift1}_{drift2}",
                f"{path}/{name}_4_{drift1}_{drift2}",
                f"{path}/{name}_5_{drift1}_{drift2}"
                ]]
    plot_drift(experiments_list, tasks, name)
                
