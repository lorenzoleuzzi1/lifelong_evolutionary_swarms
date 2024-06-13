# load log book json
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from deap import base
# TODO: make it better
def load_logbook_json(logbook_path):
    with open(logbook_path, "r") as f:
        logbook = json.load(f)
    return logbook

def plot_drift(path1, path2):
    log1 = load_logbook_json(path1)
    log2 = load_logbook_json(path2)
    len1 = len(log1["best"])
    len2 = len(log2["best"])

    best = log1["best"] + log2["best"]
    # Plot best
    plt.plot(range(len1 + len2), best, label="Best")
    # vertical line in len1
    plt.axvline(x=len1, color='g', linestyle='--')
    plt.legend()
    plt.title("Drift")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig("drift.png")
    plt.show()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Evolutionary swarm drift (change of objective).')
    # parser.add_argument('--experiment1', type=str, help=f'The name (path) of the first experiment.')
    # parser.add_argument('--experiment2', type=str, help=f'The name (path) of the second experiment DRIFT.')
    # args = parser.parse_args()
    # if not os.path.exists(args.experiment1):
    #     raise ValueError(f"Experiment {args.experiment1} does not exist.")
    # if not os.path.exists(args.experiment2):
    #     raise ValueError(f"Experiment {args.experiment2} does not exist.")
    # plot_drift(args.experiment1, args.experiment2)
    plot_drift("results/d_neat_500_50_500_8_20_99/logbook.json", "results/d_neat_500_50_500_8_20_99_drift5/logbook.json")
