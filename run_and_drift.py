from experiment import EvoSwarmExperiment, EVOLUTIONARY_ALGORITHMS
from environment import SwarmForagingEnv, BLUE, RED
from neural_controller import NeuralController
import argparse

def main(name, 
        evolutionary_algorithm, 
        steps,
        generations,
        population_size,
        n_agents, 
        n_blocks,
        distribution,
        regularization,
        lambdas,
        eval_retaining,
        seed,
        workers, 
        drifts):
    
    env = SwarmForagingEnv(n_agents = n_agents, n_blocks = n_blocks, target_color=drifts[0],
                           duration=steps, distribution=distribution)
    
    controller_deap = None
    config_path_neat = None
    if evolutionary_algorithm == "neat":
        config_path_neat = "config-feedforward.txt"
    else:
        input_dim = (env.n_types + 2 + 1) * env.n_neighbors + 2 + env.n_types - 2
        output_dim = 3
        hidden_units = [4]
        layer_sizes = [input_dim] + hidden_units + [output_dim]
        controller_deap = NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")
    
    experiment = EvoSwarmExperiment(env = env, name = name, evolutionary_algorithm=evolutionary_algorithm, 
                                    population_size=population_size, 
                                    controller_deap=controller_deap, 
                                    config_path_neat=config_path_neat, 
                                    reg_lambdas=lambdas,
                                    seed = seed)

    experiment.run(generations, n_workers = workers)
    
    for drift in drifts[1:]:
        experiment.change_objective(drift)
        experiment.run(generations, n_workers = workers,
                       eval_retaining = eval_retaining, 
                       regularization_retaining = regularization)
        
if __name__ == "__main__":
    # TODO: add as argument parrallel and drifts
    parser = argparse.ArgumentParser(description='Evolutionary swarm parameters.')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    parser.add_argument('--evo', type=str, default="neat", help=f'The name of the script to run. Must be one of: {EVOLUTIONARY_ALGORITHMS}')
    parser.add_argument('--steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--generations', type=int, default=200,help='The number of generations to run the algorithm.')
    parser.add_argument('--population', type=int, default=300,help='The size of the population for the evolutionary algorithm.')
    parser.add_argument('--agents', type=int, default=5,help='The number of agents in the arena.')
    parser.add_argument('--blocks', type=int, default=20,help='The number of blocks in the arena.')
    parser.add_argument('--distribution', type=str, default="uniform", help='The distribution of the blocks in the arena. Must be one of: uniform or biased.')
    parser.add_argument('--regularization', type=str, default=None, help='The regularization to use.')
    parser.add_argument('--lambdas', type=float, nargs="*", default=None, help='The weight regularization parameter.')
    parser.add_argument('--eval_retaining', type=str, default="top", help='The evaluation retaining strategy.')
    parser.add_argument('--seed', type=int, default=0,help='The seed for the random number generator.')
    parser.add_argument('--workers', type=int, default=1,help='The number of workers to run the algorithm.')
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
    if args.distribution not in ["uniform", "biased"]:
        raise ValueError("Distribution must be one of: uniform or biased")
    if args.regularization not in ["gd", "wp", "genetic_distance", "weight_protection"]:
        raise ValueError("Regularization must be one of: gd, wp, genetic_distance, weight_protection")
    print(args.lambdas)
    for l in args.lambdas:
        if l < 0:
            raise ValueError("Lambda must be greater than or equal to 0")
    if len(args.lambdas) != 1 and args.regularization == "gd":
        raise ValueError("Genetic distance needs 1 lambda")
    if len(args.lambdas) != 2 and args.regularization == "wp":
        raise ValueError("Weight protection needs 2 lambdas")
    if args.regularization == "gd":
        args.lambdas = {"gd" : args.lambdas[0]}
    elif args.regularization == "wp":
        args.lambdas = {"wp" : args.lambdas}
    if args.eval_retaining not in ["top", "population"]:
        raise ValueError("Evaluation retaining must be one of: top or population")
    if args.seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")
    if args.workers <= 0:
        raise ValueError("Number of workers must be greater than 0")
    
    drifts = [RED, BLUE, RED] # TODO: add as argument
    
    main(args.name, 
        args.evo, 
        args.steps,
        args.generations, 
        args.population,
        args.agents, 
        args.blocks, 
        args.distribution,
        args.regularization,
        args.lambdas,
        args.eval_retaining,
        args.seed,
        args.workers,
        drifts)