from experiment import LifelongEvoSwarmExperiment

def main(path):
    e = LifelongEvoSwarmExperiment()
    e.load(path)
    # e.run_best(on_prev_env = "top")
    # e.run_best(on_prev_env = "find_best")
    # e.run_best(on_prev_env = "random")
    # e._save_results()
    
if __name__ == '__main__':
    path = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results/evo_algos/neat_500_300_300_5_20_u/seed1/static3"
    main(path)