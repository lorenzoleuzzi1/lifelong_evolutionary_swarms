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
    # read env
    with open(f'{directory_path}/{folders[0]}/env.pkl', 'rb') as f:
        env = pickle.load(f)
    reward_threshold = env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP) # TODO: maybe put threshold in env
    # Plot all the bests lighter and the average darker
    mean_best = np.mean(bests, axis=0)
    for best in bests:
        plt.plot(best, color='blue', alpha=0.3)
    plt.plot(mean_best, color='blue', alpha=1)
    plt.axhline(y=reward_threshold, color='g', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Evolution - {alg_name}')
    plt.savefig(f'{directory_path}/{exp_name}_{alg_name}_evolution.png')
    plt.show()

plot_avg_evo('neat', 'whsim')
plot_avg_evo('cma-es', 'whsim')
plot_avg_evo('ga', 'whsim')
