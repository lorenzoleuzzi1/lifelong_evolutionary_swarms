from experiment import EvoSwarmExperiment
def main(path):
    e = EvoSwarmExperiment()
    e.load(path)
    e.run_best(on_prev_env = "top")
    e.run_best(on_prev_env = "find_best")
    e.run_best(on_prev_env = "random")
    
if __name__ == '__main__':
    path = "results/long_neat_800_400_300_5_30_u_5_drift34_drift43"
    main(path)