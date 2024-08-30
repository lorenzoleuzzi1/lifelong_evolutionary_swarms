import os
import json
import numpy as np

# read experiment.json of every folder in results
def load_experiment(path):
    file = f"{path}/experiment.json"
    if not os.path.exists(file):
        return None
    with open(f"{path}/experiment.json", "r") as f:
        experiment = json.load(f)
    return experiment

all_results_path = "results"
results = os.listdir(all_results_path)
# only the one that starts with "long"
# results = [result for result in results if result.startswith("long")]
results = [f"{all_results_path}/{result}" for result in results]

# load experiment.json
experiments = [load_experiment(result) for result in results]

# sum the correct retrives and wrong retrives
correct_retrives = 0
wrong_retrives = 0
for experiment in experiments:
    if experiment is None or "correct_retrieves" not in experiment or "wrong_retrieves" not in experiment:
        continue
    correct_retrives += experiment["correct_retrieves"]
    wrong_retrives += experiment["wrong_retrieves"]

# calculate the accuracy
accuracy = correct_retrives / (correct_retrives + wrong_retrives)
print(f"Accuracy: {accuracy}")
print(f"Correct retrives: {correct_retrives}")
print(f"Wrong retrives: {wrong_retrives}")
