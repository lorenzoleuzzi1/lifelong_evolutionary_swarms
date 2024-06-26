import argparse
from environment import SwarmForagingEnv, RED, BLUE, GREEN, YELLOW, PURPLE
from experiment import EvoSwarmExperiment, EVOLUTIONARY_ALGORITHMS
import os

colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
def main(experiment_path, new_targets, generations):
    experiment = EvoSwarmExperiment()
    experiment.load(experiment_path)
    for new_target in new_targets:
        experiment.change_objective(new_target) 
        experiment.run(generations, n_eval_forgetting=5)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Evolutionary swarm drift (change of objective).')
    # parser.add_argument('--experiment', type=str, help=f'The name (path) of the experiment.')
    # parser.add_argument('--drift', type=str, default = GREEN, help=f'The new objective for the agents.')
    # args = parser.parse_args()
    # if args.drift not in colors:
    #     raise ValueError(f"Drift must be one of: {colors}")
    # if not os.path.exists(args.experiment):
    #     raise ValueError(f"Experiment {args.experiment} does not exist.")

    # main(args.experiment, args.drift)
    main("results/uni_neat_800_300_500_8_30_32", [BLUE, RED], 300)