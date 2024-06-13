import argparse
from environment import SwarmForagingEnv, RED, BLUE, GREEN, YELLOW, PURPLE
from experiment import EvoSwarmExperiment, EVOLUTIONARY_ALGORITHMS
import os

colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
def main(experiment_path, new_target, generations):
    experiment = EvoSwarmExperiment()
    experiment.load(experiment_path)
    experiment.change_objective(new_target) 
    # TODO: change the environment to the new target
    experiment.env = SwarmForagingEnv(target_color=GREEN,
        n_agents = experiment.env.n_agents, n_blocks = experiment.env.n_blocks, seed = experiment.env.seed, duration=experiment.env.duration)
    experiment.run(generations)

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
    main("results/d_neat_500_50_500_8_20_99", GREEN, 5)