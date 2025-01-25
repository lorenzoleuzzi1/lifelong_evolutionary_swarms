from experiment import LifelongEvoSwarmExperiment
from environment import SwarmForagingEnv
import argparse
import neat
from utils import neat_sigmoid

def main(name, 
        steps,
        generations,
        population_size,
        n_agents, 
        n_blocks,
        n_envs,
        eval_retention,
        regularization,
        lambdas,
        config_path,
        seed,
        workers, 
        ):

    colors = [3, 4, 5, 6]
    env = SwarmForagingEnv(n_agents = n_agents, n_blocks = n_blocks, colors=colors,
                           target_color=3, duration=steps,
                           season_colors=[3,4])

    # Set configuration file
    config_neat = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    config_neat.genome_config.add_activation('neat_sigmoid', neat_sigmoid)
    config_neat.pop_size = population_size
    obs_example = env.reset(seed=seed)[0]
    config_neat.genome_config.num_inputs = len(env.process_observation(obs_example)[0])
    config_neat.genome_config.input_keys = [-i - 1 for i in range(config_neat.genome_config.num_inputs)]
    
    experiment = LifelongEvoSwarmExperiment(env = env, name = name, 
                                    population_size=population_size, 
                                    config_neat=config_neat, 
                                    n_envs=n_envs,
                                    seed=seed,
                                    n_workers = workers)
    # Season 1
    experiment.run(generations)
    
    # Season 2
    experiment.drift([5,6], 5)
    experiment.run(generations,
                    eval_retention=eval_retention, 
                    regularization_type=regularization,
                    regularization_coefficient=lambdas)
    # Season 3
    experiment.drift([3,4], 3)
    experiment.run(generations,
                    eval_retention=eval_retention,
                    regularization_type=regularization,
                    regularization_coefficient=lambdas)
    

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evolutionary swarm parameters.')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    parser.add_argument('--steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--generations', type=int, default=200,help='The number of generations to run the algorithm.')
    parser.add_argument('--population', type=int, default=300,help='The size of the population for the evolutionary algorithm.')
    parser.add_argument('--agents', type=int, default=5,help='The number of agents in the arena.')
    parser.add_argument('--blocks', type=int, default=20,help='The number of blocks in the arena.')
    parser.add_argument('--evals', type=int, default=1, help='Number of environments to evaluate the fitness.')
    parser.add_argument('--regularization', type=str, default=None, help='The type regularization to use.')
    parser.add_argument('--lambdas', type=float, nargs="*", default=None, help='The weight regularization parameter.')
    parser.add_argument('--eval_retention', type=str, nargs="*", default=None, help='The evaluation retention strategy.')
    parser.add_argument('--config', type=str, default="config-feedforward.txt", help='The configuration file for NEAT.')
    parser.add_argument('--seed', type=int, default=0,help='The seed for the random number generator.')
    parser.add_argument('--workers', type=int, default=1, help='The number of workers to run the algorithm.')
    args = parser.parse_args()
    
    if args.steps <= 0:
        raise ValueError("Number of steps must be greater than 0")
    if args.generations <= 0:
        raise ValueError("Number of generations must be greater than 0")
    if args.population <= 0:
        raise ValueError("Population size must be greater than 0")
    if args.agents <= 0:
        raise ValueError("Number of agents must be greater than 0")
    if args.blocks <= 0:
        raise ValueError("Number of blocks must be greater than 0")
    if args.evals <= 0:
        raise ValueError("Number of environments must be greater than 0")
    if args.regularization is not None and args.regularization not in ["gd", "wp", "genetic_distance", "weight_protection", "functional"]:
        raise ValueError("Regularization must be one of: gd, wp, genetic_distance, weight_protection, functional")
    if args.eval_retention is not None:
        for e in args.eval_retention:
            if e not in ["top", "population", "pop"]:
                raise ValueError("Evaluation retention must be one of: top or population / pop")
    if args.lambdas is not None:
        for l in args.lambdas:
            if l < 0:
                raise ValueError("Lambda must be greater than or equal to 0")
        if len(args.lambdas) != 1 and args.regularization == "gd":
            raise ValueError("Genetic distance needs 1 lambda")
        if len(args.lambdas) != 1 and args.regularization == "functional":
            raise ValueError("Functional needs 1 lambda")
        if len(args.lambdas) != 2 and args.regularization == "wp":
            raise ValueError("Weight protection needs 2 lambdas")
        args.lambdas = args.lambdas[0]
    if args.seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")
    if args.workers <= 0:
        raise ValueError("Number of workers must be greater than 0")
    
    #cProfile.run("main(args.name, args.evo, args.steps, args.generations, args.population, args.agents, args.blocks, args.colors, args.distribution, args.targets, args.n_env, args.eval_retention, args.regularization, args.lambdas, args.seed, args.workers)")
    main(args.name, 
        args.steps,
        args.generations, 
        args.population,
        args.agents, 
        args.blocks, 
        args.evals,
        args.eval_retention,
        args.regularization,
        args.lambdas,
        args.config,
        args.seed,
        args.workers,
        )