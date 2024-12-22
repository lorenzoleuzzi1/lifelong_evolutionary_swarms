from experiment import LifelongEvoSwarmExperiment, EVOLUTIONARY_ALGORITHMS
from environment import SwarmForagingEnv, BLUE, RED
from neural_controller import NeuralController
import argparse
import neat
from utils import neat_sigmoid
import cProfile

def main(name, 
        evolutionary_algorithm, 
        steps,
        generations,
        population_size,
        n_agents, 
        n_blocks,
        n_colors,
        distribution,
        drifts,
        n_env,
        eval_retention,
        regularization,
        lambdas,
        seed,
        workers, 
        ):
    
    env = SwarmForagingEnv(n_agents = n_agents, n_blocks = n_blocks, n_colors=n_colors,
                           target_color=drifts[0], duration=steps, distribution=distribution)
    initial_state, _ = env.reset(seed=seed)
    
    controller_deap = None
    config_neat = None
    if evolutionary_algorithm == "neat":
        config_path_neat = "config-feedforward.txt"
        # Set configuration file
        config_neat = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_neat)
        config_neat.genome_config.add_activation('neat_sigmoid', neat_sigmoid)
        config_neat.pop_size = population_size
        obs_example = env.reset(seed=seed)[0]
        config_neat.genome_config.num_inputs = len(env.process_observation(obs_example)[0])
        config_neat.genome_config.input_keys = [-i - 1 for i in range(config_neat.genome_config.num_inputs)]
    else:
        input_dim = len(env.process_observation(initial_state)[0])
        output_dim = 3
        layer_sizes = [input_dim] + [output_dim]
        controller_deap = NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")
    
    experiment = LifelongEvoSwarmExperiment(env = env, name = name, evolutionary_algorithm=evolutionary_algorithm, 
                                    population_size=population_size, 
                                    controller_deap=controller_deap, 
                                    config_neat=config_neat, 
                                    reg_lambdas=lambdas,
                                    n_env = n_env,
                                    seed = seed,
                                    n_workers = workers)

    experiment.run(generations)
    
    for drift in drifts[1:]:
        experiment.drift(drift)
        experiment.run(generations,
                       eval_retention = eval_retention, 
                       regularization_retention = regularization)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evolutionary swarm parameters.')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    parser.add_argument('--evo', type=str, default="neat", help=f'The name of the script to run. Must be one of: {EVOLUTIONARY_ALGORITHMS}')
    parser.add_argument('--steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--generations', type=int, default=200,help='The number of generations to run the algorithm.')
    parser.add_argument('--population', type=int, default=300,help='The size of the population for the evolutionary algorithm.')
    parser.add_argument('--agents', type=int, default=5,help='The number of agents in the arena.')
    parser.add_argument('--blocks', type=int, default=20,help='The number of blocks in the arena.')
    parser.add_argument('--colors', type=int, default=5,help='The number of colors available in the arena.')
    parser.add_argument('--distribution', type=str, default="uniform", help='The distribution of the blocks in the arena. Must be one of: uniform or biased.')
    parser.add_argument('--targets', type=int, nargs="*", default=[3], help='The targets and drifts (change of target color) to apply.')
    parser.add_argument('--n_env', type=int, default=1, help='Number of environments to evaluate the fitness.')
    parser.add_argument('--regularization', type=str, default=None, help='The regularization to use.')
    parser.add_argument('--lambdas', type=float, nargs="*", default=None, help='The weight regularization parameter.')
    parser.add_argument('--eval_retention', type=str, nargs="*", default=None, help='The evaluation retention strategy.')
    parser.add_argument('--seed', type=int, default=0,help='The seed for the random number generator.')
    parser.add_argument('--workers', type=int, default=1, help='The number of workers to run the algorithm.')
    args = parser.parse_args()
    
    if args.evo not in EVOLUTIONARY_ALGORITHMS:
        raise ValueError(f"Script must be one of {EVOLUTIONARY_ALGORITHMS}")
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
    if args.colors <= 0:
        raise ValueError("Number of colors must be greater than 0")
    if args.distribution not in ["uniform", "biased"]:
        raise ValueError("Distribution must be one of: uniform or biased")
    if args.n_env <= 0:
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
    if args.regularization == "gd":
        args.lambdas = {"gd" : args.lambdas[0]}
    elif args.regularization == "wp":
        args.lambdas = {"wp" : args.lambdas}
    elif args.regularization == "functional":
        args.lambdas = {"functional" : args.lambdas[0]}
    if args.seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")
    if args.workers <= 0:
        raise ValueError("Number of workers must be greater than 0")
    if args.evo != "neat" and len(args.targets) > 1:
        raise ValueError(f"Drifts are implemented only using NEAT, got {args.evo}.")
    #cProfile.run("main(args.name, args.evo, args.steps, args.generations, args.population, args.agents, args.blocks, args.colors, args.distribution, args.targets, args.n_env, args.eval_retention, args.regularization, args.lambdas, args.seed, args.workers)")
    main(args.name, 
        args.evo, 
        args.steps,
        args.generations, 
        args.population,
        args.agents, 
        args.blocks, 
        args.colors,
        args.distribution,
        args.targets,
        args.n_env,
        args.eval_retention,
        args.regularization,
        args.lambdas,
        args.seed,
        args.workers,
        )