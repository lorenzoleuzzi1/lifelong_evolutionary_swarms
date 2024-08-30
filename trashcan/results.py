import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import environment
# TODO read info and avg blocks retrieved
def plot_avg_evo(alg_name, exp_name):
    # Directory where the files are located
    directory_path = 'results'

    # List all files in the directory
    all_files = os.listdir(directory_path)
    # Filter files that contain 'neat' in their names
    folders = [f for f in all_files if alg_name in f and exp_name in f]

    # Filter files that contain 'neat' in their names
    bests = []
    for folder in folders:
        # read logbook.json
        with open(f'{directory_path}/{folder}/logbook.json', 'r') as f:
            logbook = json.load(f)
            bests.append(logbook['best'])
    
    # Plot all the bests lighter and the average darker
    mean_best = np.mean(bests, axis=0)
    for best in bests:
        plt.plot(best, color='blue', alpha=0.3)
    plt.plot(mean_best, color='blue', alpha=1)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Evolution - {alg_name}')
    plt.savefig(f'{directory_path}/{exp_name}_{alg_name}_evolution.png')
    plt.show()

plot_avg_evo('neat', 'uni')
plot_avg_evo('cma-es', 'uni')
plot_avg_evo('ga', 'uni')
plot_avg_evo('evostick', 'uni')
