import os
import json
IMPORTANCE_NOT_DROPPING_ON_CURRENT = 0.5
BASELINE_U_CURRENT_DRIFT34 = 12.4
BASELINE_U_CURRENT_DRIFT43 = 13.0
BASELINE_B_CURRENT_DRIFT34 = 30.1
BASELINE_B_CURRENT_DRIFT43 = 33.0
# We want at least half for the current
BASELINE_U_CURRENT_HALF_DRIFT34 = BASELINE_U_CURRENT_DRIFT34 * IMPORTANCE_NOT_DROPPING_ON_CURRENT
BASELINE_U_CURRENT_HALF_DRIFT43 = BASELINE_U_CURRENT_DRIFT43 * IMPORTANCE_NOT_DROPPING_ON_CURRENT
BASELINE_B_CURRENT_HALF_DRIFT34 = BASELINE_B_CURRENT_DRIFT34 * IMPORTANCE_NOT_DROPPING_ON_CURRENT
BASELINE_B_CURRENT_HALF_DRIFT43 = BASELINE_B_CURRENT_DRIFT43 * IMPORTANCE_NOT_DROPPING_ON_CURRENT

BASELINE_U_RETENTION_DRIFT34 = -0.3
BASELINE_U_RETENTION_DRIFT43 = 1.0
BASELINE_B_RETENTION_DRIFT34 = 3.5
BASELINE_B_RETENTION_DRIFT43 = 6.5

def load_experiment_json(path):
    experiment_path = f"{path}/experiment.json"
    with open(experiment_path, "r") as f:
        experiment = json.load(f)
    return experiment

# read experiments folder
results_path = os.path.abspath("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong evolutionary swarms/results/drift_search_coarse/results")

reg = "gd"
distribution = "_u_"
drift = "drift34" # or drift43
seeds = range(10)
seeds_info = {}

# seed info := seed : (trashold_accepting, best_current, best_retantion, lambda)
if drift == "drift34" and distribution == "_u_":
    for seed in seeds:
        seeds_info[seed] = (BASELINE_U_CURRENT_HALF_DRIFT34, None, BASELINE_U_RETENTION_DRIFT34, None)
if drift == "drift43" and distribution == "_u_":
    for seed in seeds:
        seeds_info[seed] = (BASELINE_U_CURRENT_HALF_DRIFT43, None, BASELINE_U_RETENTION_DRIFT43, None)
if drift == "drift34" and distribution == "_b_":
    for seed in seeds:
        seeds_info[seed] = (BASELINE_B_CURRENT_HALF_DRIFT34, None, BASELINE_B_RETENTION_DRIFT34, None)
if drift == "drift43" and distribution == "_b_":
    for seed in seeds:
        seeds_info[seed] = (BASELINE_U_CURRENT_HALF_DRIFT43, None, BASELINE_B_RETENTION_DRIFT43, None)

# order files by name
allfiles = os.listdir(results_path)
allfiles.sort()
for filename in allfiles:
    # Check if the filename matches the pattern
    if reg in filename and distribution in filename and filename.endswith(drift):

        experiment = load_experiment_json(f"{results_path}/{filename}")
        seed = experiment["seed"]
        lambda_env = experiment["regularization_lambdas"][reg]
        best_fitness_retaining_top = experiment["best_fitness_retaining_top"]
        best_fitness_current = experiment["best_fitness"]
        
        # We want to find the lambda that improve the baseline in retention while not dropping to much (at least half) in current
        if best_fitness_current > seeds_info[seed][0] and best_fitness_retaining_top > seeds_info[seed][2]:
            seeds_info[seed] = (seeds_info[seed][0], best_fitness_current, best_fitness_retaining_top, lambda_env)
    
print(seeds_info)
                


