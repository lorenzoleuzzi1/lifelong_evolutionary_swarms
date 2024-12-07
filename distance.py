import os
import pickle as pkl
import numpy as np
import json

results_dir = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results/results"
reg = "wp"
dir = f"{results_dir}/reg_{reg}"
distribution = "u"
baseline_folder = "0" if reg == "gd" else "0_0"
 

distances = {}

baseline_per_seed = {}
seed_folders = []
for lambda_folder in os.listdir(dir):
    if os.path.isdir(f"{dir}/{lambda_folder}"):
        if lambda_folder == baseline_folder:
            for distribution_folder in os.listdir(f"{dir}/{lambda_folder}"):
                if distribution in distribution_folder:
                    # HERE JUST SAVE BASELINES
                    for seed_folder in os.listdir(f"{dir}/{lambda_folder}/{distribution_folder}"):
                        
                        first_evo_flag = True
                        distances_per_evolution = []
                        
                        if os.path.isdir(f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}"):
                            if seed_folder not in seed_folders:
                                seed_folders.append(seed_folder)
                            
                            for evolution_folder in os.listdir(f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}"):
                                if evolution_folder == "static3": 
                                    genome_path = f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}/{evolution_folder}/best_genome.pkl"
                                    config_path = f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}/{evolution_folder}/neat_config.pkl"
                                    
                                    with open(genome_path, 'rb') as f:
                                        try:
                                            net = pkl.load(f)
                                        except pkl.UnpicklingError as e:
                                            print(f"Error unpickling file: {e}")
                                        except Exception as e:
                                            print(f"An error occurred: {e}")
                                    with open(config_path, "rb") as f:
                                        config = pkl.load(f)
                                    config.compatibility_weight_coefficient = 0.6
                                    config.compatibility_disjoint_coefficient = 1.0

                                    baseline_per_seed[seed_folder] = (net, config)

print(baseline_per_seed)
distances_baseline = {}
print(seed_folders)
for i in range(len(baseline_per_seed)):
    for j in range(i+1, len(baseline_per_seed)):
        if reg == "gd":
            d = baseline_per_seed[seed_folders[i]][0].distance(baseline_per_seed[seed_folders[j]][0], baseline_per_seed[seed_folders[j]][1])
        else:
            penalty_wp1 = 0.0
            penalty_wp2 = 0.0
            for c in baseline_per_seed[seed_folders[i]][0].connections:
                if c in baseline_per_seed[seed_folders[j]][0].connections:
                    penalty_wp1 += (baseline_per_seed[seed_folders[i]][0].connections[c].weight - baseline_per_seed[seed_folders[j]][0].connections[c].weight) **2
                else:
                    penalty_wp2 += baseline_per_seed[seed_folders[j]][0].connections[c].weight ** 2
            d = (penalty_wp1, penalty_wp2)

        distances_baseline[f"{seed_folders[i]}, {seed_folders[j]}"] = d

print(distances_baseline)
# Save json
with open(f"{dir}/distances_firstevo_{reg}_{distribution}.json", "w") as f:
    json.dump(distances_baseline, f, indent=4)

for lambda_folder in os.listdir(dir):
    if os.path.isdir(f"{dir}/{lambda_folder}"):
        for distribution_folder in os.listdir(f"{dir}/{lambda_folder}"):
            if distribution in distribution_folder:
                # SEEDS
                distances_per_seed = {}
                for seed_folder in os.listdir(f"{dir}/{lambda_folder}/{distribution_folder}"):
                    
                    first_evo_flag = True
                    distances_per_evolution = []
                    if os.path.isdir(f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}"):
                        nets_evo = {}
                        for evolution_folder in os.listdir(f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}"):
                            if os.path.isdir(f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}/{evolution_folder}"):
                                genome_path = f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}/{evolution_folder}/best_genome.pkl"
                                config_path = f"{dir}/{lambda_folder}/{distribution_folder}/{seed_folder}/{evolution_folder}/neat_config.pkl"
                                
                                with open(genome_path, 'rb') as f:
                                    try:
                                        net = pkl.load(f)
                                    except pkl.UnpicklingError as e:
                                        print(f"Error unpickling file: {e}")
                                    except Exception as e:
                                        print(f"An error occurred: {e}")
                                with open(config_path, "rb") as f:
                                    config = pkl.load(f)

                                config.compatibility_weight_coefficient = 0.6
                                config.compatibility_disjoint_coefficient = 1.0
                                nets_evo[evolution_folder] = (net, config)

                        if reg == "gd":
                            d_1_2 = nets_evo["static3"][0].distance(nets_evo["static3_drift34"][0], nets_evo["static3_drift34"][1])
                            d_2_3 = nets_evo["static3_drift34"][0].distance(nets_evo["static3_drift34_drift43"][0], nets_evo["static3_drift34_drift43"][1])
                        else:
                            penalty_wp1 = 0.0
                            penalty_wp2 = 0.0
                            for c in nets_evo["static3"][0].connections:
                                if c in nets_evo["static3_drift34"][0].connections:
                                    penalty_wp1 += (nets_evo["static3"][0].connections[c].weight - nets_evo["static3_drift34"][0].connections[c].weight) **2
                                else:
                                    penalty_wp2 += nets_evo["static3_drift34"][0].connections[c].weight ** 2
                            
                            d_1_2 = (penalty_wp1, penalty_wp2)

                            penalty_wp1 = 0.0
                            penalty_wp2 = 0.0
                            for c in nets_evo["static3_drift34"][0].connections:
                                if c in nets_evo["static3_drift34_drift43"][0].connections:
                                    penalty_wp1 += (nets_evo["static3_drift34"][0].connections[c].weight - nets_evo["static3_drift34_drift43"][0].connections[c].weight) **2
                                else:
                                    penalty_wp2 += nets_evo["static3_drift34_drift43"][0].connections[c].weight ** 2
                            
                            d_2_3 = (penalty_wp1, penalty_wp2)

                        distances_per_evolution.append(d_1_2)
                        distances_per_evolution.append(d_2_3)
                        distances_per_seed[seed_folder] = distances_per_evolution

                                
                            # if first_evo_flag:
                            #     reference_net = net
                            #     baseline_net = net
                            #     first_evo_flag = False
                            # else:
                            #     if reg == "gd":
                            #         config.compatibility_weight_coefficient = 0.6
                            #         config.compatibility_disjoint_coefficient = 1.0
                            #         d = reference_net.distance(net, config)
                            #     else:
                            #         penalty_wp1 = 0.0
                            #         penalty_wp2 = 0.0
                            #         for c in net.connections:
                            #             if c in reference_net.connections:
                            #                 penalty_wp1 += (reference_net.connections[c].weight - net.connections[c].weight) **2
                            #             else:
                            #                 penalty_wp2 += net.connections[c].weight ** 2

                            #         d = (penalty_wp1, penalty_wp2)
                                
                            #     distances_per_evolution.append(d)
                            #     reference_net = net
                            
                            # distances_per_seed[seed_folder] = distances_per_evolution

                distances[lambda_folder] = distances_per_seed

# save the distances in json
with open(f"{dir}/distances_{reg}_{distribution}.json", "w") as f:
    json.dump(distances, f, indent=4)

# with open("/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results/results/reg_gd/distances_u.json", "r") as f:
#     distances = json.load(f)

# Create a dictionary to store the mean values for each seed
std_devs_evo2 = {}
means_evo2 = {}
std_devs_evo3 = {}
means_evo3 = {}
# Iterate over each entry in the data
for key, seed_data in distances.items():
    d_evo2 = []
    d_evo3 = []
    for seed, values in seed_data.items():
        d_evo2.append(values[0])
        d_evo3.append(values[1])
    
    std_devs_evo2[key] = np.std(d_evo2)
    means_evo2[key] = np.mean(d_evo2)
    std_devs_evo3[key] = np.std(d_evo3)
    means_evo3[key] = np.mean(d_evo3)

print("Means evo2: ", means_evo2)
print("Std Devs evo2: ", std_devs_evo2)
print("Means evo3: ", means_evo3)
print("Std Devs evo3: ", std_devs_evo3)

                
                
                     