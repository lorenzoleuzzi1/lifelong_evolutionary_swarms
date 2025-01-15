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
        colors,
        drifts,
        n_envs,
        eval_retention,
        regularization,
        lambdas,
        seed,
        workers, 
        ):

    env = SwarmForagingEnv(n_agents = n_agents, n_blocks = n_blocks, colors=colors,
                           target_color=drifts[0], duration=steps)
    # initial_state, _ = env.reset(seed=seed)
    
    config_path_neat = "config-feedforward.txt"
    # Set configuration file
    config_neat = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_neat)
    config_neat.genome_config.add_activation('neat_sigmoid', neat_sigmoid)
    config_neat.pop_size = population_size
    obs_example = env.reset(seed=seed)[0]
    config_neat.genome_config.num_inputs = len(env.process_observation(obs_example)[0])
    config_neat.genome_config.input_keys = [-i - 1 for i in range(config_neat.genome_config.num_inputs)]
    
    experiment = LifelongEvoSwarmExperiment(env = env, name = name, 
                                    population_size=population_size, 
                                    config_neat=config_neat, 
                                    # reg_lambdas=lambdas, # TODO: put lambad in .run
                                    n_envs=n_envs,
                                    seed=seed,
                                    n_workers = workers)

    experiment.run(generations)
    
    for drift in drifts[1:]:
        experiment.drift(drift)
        experiment.run(generations,
                       eval_retention = eval_retention, #TODO: maybe rename
                       regularization_type = regularization,
                       regularization_coefficient=lambdas)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evolutionary swarm parameters.')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    parser.add_argument('--steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--generations', type=int, default=200,help='The number of generations to run the algorithm.')
    parser.add_argument('--population', type=int, default=300,help='The size of the population for the evolutionary algorithm.')
    parser.add_argument('--agents', type=int, default=5,help='The number of agents in the arena.')
    parser.add_argument('--blocks', type=int, default=20,help='The number of blocks in the arena.')
    parser.add_argument('--colors', type=int, nargs="*", default=[3,4], help='The colors of the blocks in the arena. 3: red, 4: blue, 5: green, 6: yellow, 7: purple.')
    parser.add_argument('--rate_target', type=str, default=0.5, help='The rate of the target blocks in the arena.')
    parser.add_argument('--targets', type=int, nargs="*", default=[3], help='The targets and drifts (change of target color) to apply. 3: red, 4: blue, 5: green, 6: yellow, 7: purple.')
    parser.add_argument('--evals', type=int, default=1, help='Number of environments to evaluate the fitness.')
    parser.add_argument('--regularization', type=str, default=None, help='The type regularization to use.')
    parser.add_argument('--lambdas', type=float, nargs="*", default=None, help='The weight regularization parameter.')
    parser.add_argument('--eval_retention', type=str, nargs="*", default=None, help='The evaluation retention strategy.')
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
    for color in args.colors:
        if color not in [3, 4, 5, 6, 7]:
            raise ValueError("Color must be one of: 3, 4, 5, 6, 7 (representing the color of the blocks red, blue, green, yellow, purple)")
    if args.evals <= 0:
        raise ValueError("Number of environments must be greater than 0")
    if args.regularization is not None and args.regularization not in ["gd", "wp", "genetic_distance", "weight_protection", "functional"]:
        raise ValueError("Regularization must be one of: gd, wp, genetic_distance, weight_protection, functional")
    if args.targets is not None:
        for t in args.targets:
            if t not in [3, 4, 5, 6, 7]:
                raise ValueError("Drift must be one of: 3, 4, 5, 6, 7 (representing the color of the target red, blue, green, yellow, purple)")
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
        args.colors,
        args.targets,
        args.evals,
        args.eval_retention,
        args.regularization,
        args.lambdas[0], #TODO
        args.seed,
        args.workers,
        )