# load log book json
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from deap import base

def load_logbook_json(path):
    logbook_path = f"{path}/logbook.json"
    with open(logbook_path, "r") as f:
        logbook = json.load(f)
    return logbook

def plot_drift(paths):
    
    logs = [load_logbook_json(path) for path in paths]
    bests = [log["best"] for log in logs]
    forgetting = [log["forgetting"] for log in logs[1:]]
    gen_drifts = [0]
    for i in range(1, len(bests)):
        gen_drifts.append(len(bests[i-1]) + gen_drifts[i-1])
    bests = np.array(bests).flatten()
    print(f"Average Anytime Fitness: {np.mean(bests)}")
    forgetting = np.array(forgetting).flatten()
    # Plot best
    plt.plot(range(len(bests)), bests, label="Best")
    # Plot forgetting
    plt.plot(range(gen_drifts[1], gen_drifts[1]+ len(forgetting)), forgetting, label="Retaining", color='r', alpha=0.4)
    for i, l in enumerate(gen_drifts[1:]):
        if i == 0:
            plt.axvline(x=l, color='g', linestyle='--', label="Drift")
        else:
            plt.axvline(x=l, color='g', linestyle='--')
    plt.legend()
    plt.title("Drift")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig("drift-okb.png")
    plt.show()


def plot_drift(paths, name):
    """
    paths must be in the format:
    [[path1, path2, ...], [pathdrift1, pathdrift2, ...], [pathdriftdrift1, pathdriftdrift2, ...], ...]
    """
    counter_gens = 0
    baf = [] # best anytime fitness
    aeef = [] # average end evolution fitness

    for i in range(len(paths)):
        # Iterate thru the drifts
        log_drift = []
        for j in range(len(paths[i])):
            # Iterates thru the experiments instances
            log = load_logbook_json(paths[i][j])
            log_drift.append(log)
        bests = [log["best"] for log in log_drift]
        avg_bests = np.mean(bests, axis=0)
        current_gen = counter_gens
        counter_gens += len(avg_bests)
        baf.extend(avg_bests)
        aeef.append(avg_bests[-1])
        # Plot best
        if i == 0:
            plt.plot(range(current_gen, counter_gens), avg_bests, color='blue', label="Avg Best")
        else:
            plt.plot(range(current_gen, counter_gens), avg_bests, color='blue')
        for best in bests:
            plt.plot(range(current_gen, counter_gens), best, color='blue', alpha=0.2)
        aaf = []
        # Plot forgetting
        if i != 0:
            forgetting = [log["retaining"] for log in log_drift[1:]]
            avg_forgetting = np.mean(forgetting, axis=0)
            if i == 1:
                plt.plot(range(current_gen, counter_gens), avg_forgetting, label="Avg Retaining (Top)", color='r')
                plt.axvline(x=current_gen, color='g', linestyle='--', label="Drift")
            else:
                plt.plot(range(current_gen, counter_gens), avg_forgetting, color='r')
                plt.axvline(x=current_gen, color='g', linestyle='--')
            for retaining in forgetting:
                plt.plot(range(current_gen, counter_gens), retaining, color='red', alpha=0.2)
    baf = np.array(baf).flatten()
    print(f"Average Anytime Fitness: {np.mean(baf)}")
    print(f"Average End Evolution Fitness: {np.mean(aeef)}")        
    plt.legend()
    plt.title("Drift")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig(f"{name}.png")
    plt.show()

if __name__ == "__main__":
    
    # plot_drift([["results/baselineblueb_neat_800_100_300_5_30_1",
    #             "results/baselineblueb_neat_800_100_300_5_30_2",
    #             "results/baselineblueb_neat_800_100_300_5_30_3",
    #             "results/baselineblueb_neat_800_100_300_5_30_4",
    #             "results/baselineblueb_neat_800_100_300_5_30_5"]]
    #             ,
    #             "bbbb")
    
    plot_drift([["results/driftlfind_neat_800_200_300_5_30_u_1",
                "results/driftlfind_neat_800_200_300_5_30_u_2",
                "results/driftlfind_neat_800_200_300_5_30_u_3",
                "results/driftlfind_neat_800_200_300_5_30_u_4",
                "results/driftlfind_neat_800_200_300_5_30_u_5"]
                ,
                ["results/driftlfind_neat_800_200_300_5_30_u_1_drift34",
                "results/driftlfind_neat_800_200_300_5_30_u_2_drift34",
                "results/driftlfind_neat_800_200_300_5_30_u_3_drift34",
                "results/driftlfind_neat_800_200_300_5_30_u_4_drift34",
                "results/driftlfind_neat_800_200_300_5_30_u_5_drift34"]
                ,
                ["results/driftlfind_neat_800_200_300_5_30_u_1_drift34_drift43",
                "results/driftlfind_neat_800_200_300_5_30_u_2_drift34_drift43",
                "results/driftlfind_neat_800_200_300_5_30_u_3_drift34_drift43",
                "results/driftlfind_neat_800_200_300_5_30_u_4_drift34_drift43",
                "results/driftlfind_neat_800_200_300_5_30_u_5_drift34_drift43"]],
                "longeru")
                
