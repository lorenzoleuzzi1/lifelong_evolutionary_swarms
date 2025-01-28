import pickle as pkl
import json
import environment    

def main(path, target):
    # Load the population
    with open(path + "/population.pkl", "rb") as f:
        pop = pkl.load(f)
    # with open(f"{path}/neat_config.pkl", "rb") as f:
    #     config_neat = pkl.load(f)
    with open(f"{path}/info.json", "r") as f:
        info_exp = json.load(f)
    
    env = environment.SwarmForagingEnv(n_agents = info_exp["n_agents"], n_blocks = info_exp["n_blocks"], colors=[3, 4, 5, 6],
                                    season_colors=info_exp["season_colors"],
                            target_color=info_exp["target_color"], duration=info_exp["episode_duration"])
    if target == 3:
        env.change_season([3, 4], 3)
    if target == 5:
        env.change_season([5, 6], 5)
    
    found = False
    for id, genome in list(pop.population.items()):
        if info_exp[f"id_retention_pop_{target}"] == id:
            print("Found in pop")
            found = True
            break
    if not found:
        print("Not found in pop")

if __name__ == '__main__':
    path = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results_gecco4/snz_baselines/neat_500_200_300_5_20_10/seed17/static3_drift55"
    main(path, 3)
    path = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results_gecco4/snz_baselines/neat_500_200_300_5_20_10/seed17/static3_drift55_drift33"
    main(path, 5)