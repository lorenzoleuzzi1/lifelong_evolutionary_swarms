# load log book json
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from deap import base

def load_logbook_json(logbook_path):
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
    plot_drift(["results/driftcheckb_neat_800_100_300_5_30_69/logbook.json",
                "results/driftcheckb_neat_800_100_300_5_30_69_drift34/logbook.json",
                "results/driftcheckb_neat_800_100_300_5_30_69_drift34_drift43/logbook.json"]
                )
