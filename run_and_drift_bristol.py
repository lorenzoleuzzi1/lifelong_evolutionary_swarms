from experiment import EvoSwarmExperiment, EVOLUTIONARY_ALGORITHMS
from environment import SwarmForagingEnv, BLUE, RED
from neural_controller import NeuralController
import argparse
import numpy as np

def main(name, 
        steps,
        generations,
        population_size,
        regularization,
        lambdas,
        eval_retaining,
        seed,
        workers, 
        drifts):
    size = 16
    sensor_range = 3 # should be 4 but 3 to be sure
    sensitivity = 0.5
    max_wheel_speed = 2
    env = SwarmForagingEnv(n_agents = 3, n_blocks = 7, target_color=drifts[0],
                           duration=steps, size = size, max_retrieves=3,
                            sensor_range = sensor_range,
                            sensitivity = sensitivity,
                            max_wheel_velocity = max_wheel_speed,
                           repositioning=False)
    
    initial_setting_bristol = {}
    initial_setting_bristol['blocks'] = np.array([
        [6.6, 2.4],  # Carrier 100
        [5.1, 6.9],  # Carrier 101
        [6.6, 9.0],  # Carrier 102
        [9.6, 9.6],  # Carrier 103
        [8.1, 6.9],  # Carrier 104
        [9.9, 4.2],  # Carrier 105
        [6.9, 4.5]   # Carrier 106
    ])
    initial_setting_bristol['colors'] = np.array([3, 3, 5, 4, 3, 5, 4])
    initial_setting_bristol['agents'] =  np.array([
        [1.5, 3.0],  # Agent 0
        [1.5, 6.0],  # Agent 1
        [1.5, 9.0]   # Agent 2
    ])
    initial_setting_bristol['headings'] = np.array([180, 180, 180])
    config_path_neat = "config-feedforward.txt"
    
    experiment = EvoSwarmExperiment(env = env, name = name, evolutionary_algorithm="neat", 
                                    population_size=population_size, 
                                    config_path_neat=config_path_neat, 
                                    reg_lambdas=lambdas,
                                    env_initial_state=initial_setting_bristol,
                                    seed = seed)

    experiment.run(generations, n_workers = workers)
    
    # From red to blue
    initial_setting_bristol['colors'] = np.array([4, 4, 3, 5, 4, 3, 5])
    experiment.change_objective(drifts[1], initial_setting_bristol)
    experiment.run(generations, n_workers = workers,
                    eval_retaining = eval_retaining, 
                    regularization_retaining = regularization)
    
    initial_setting_bristol['colors'] = np.array([3, 3, 5, 4, 3, 5, 4])
    experiment.change_objective(drifts[2], initial_setting_bristol)
    experiment.run(generations, n_workers = workers,
                    eval_retaining = eval_retaining, 
                    regularization_retaining = regularization)
            
if __name__ == "__main__":
    # TODO: add as argument parrallel and drifts
    parser = argparse.ArgumentParser(description='Evolutionary swarm parameters.')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    # parser.add_argument('--evo', type=str, default="neat", help=f'The name of the script to run. Must be one of: {EVOLUTIONARY_ALGORITHMS}')
    parser.add_argument('--steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--generations', type=int, default=100,help='The number of generations to run the algorithm.')
    parser.add_argument('--population', type=int, default=100,help='The size of the population for the evolutionary algorithm.')
    # parser.add_argument('--agents', type=int, default=5,help='The number of agents in the arena.')
    # parser.add_argument('--blocks', type=int, default=20,help='The number of blocks in the arena.')
    # parser.add_argument('--distribution', type=str, default="uniform", help='The distribution of the blocks in the arena. Must be one of: uniform or biased.')
    parser.add_argument('--regularization', type=str, default=None, help='The regularization to use.')
    parser.add_argument('--lambdas', type=float, nargs="*", default=None, help='The weight regularization parameter.')
    parser.add_argument('--eval_retaining', type=str, default="top", help='The evaluation retaining strategy.')
    parser.add_argument('--seed', type=int, default=0,help='The seed for the random number generator.')
    parser.add_argument('--workers', type=int, default=1,help='The number of workers to run the algorithm.')
    args = parser.parse_args()
    
    if args.steps <= 0:
        raise ValueError("Number of steps must be greater than 0")
    if args.generations <= 0:
        raise ValueError("Number of generations must be greater than 0")
    if args.population <= 0:
        raise ValueError("Population size must be greater than 0")
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
        args.steps,
        args.generations, 
        args.population,
        args.regularization,
        args.lambdas,
        args.eval_retaining,
        args.seed,
        args.workers,
        drifts)