from experiment import EvoSwarmExperiment, EVOLUTIONARY_ALGORITHMS
from environment import SwarmForagingEnv
from neural_controller import NeuralController
import argparse

def main(name, 
        script, 
        steps,
        generations,
        population_size,
        n_agents, 
        n_blocks,
        seed):
    
    env = SwarmForagingEnv(n_agents = n_agents, n_blocks = n_blocks, seed = seed, duration=steps)
    
    
    controller_deap = None
    config_path_neat = None
    if script == "neat":
        config_path_neat = "config-feedforward.txt"
    else:
        input_dim = (env.n_types + 2 + 1) * env.n_neighbors + 2 + env.n_types - 2
        output_dim = 3
        hidden_units = [16]
        layer_sizes = [input_dim] + hidden_units + [output_dim]
        controller_deap = NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")

    experiment = EvoSwarmExperiment(env = env, name = name, evolutionary_algorithm=script, population_size=population_size, 
                                    controller_deap=controller_deap, config_path_neat=config_path_neat)
    experiment.run(generations)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary swarm parameters.')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    parser.add_argument('--evo', type=str, default="neat", help=f'The name of the script to run. Must be one of: {EVOLUTIONARY_ALGORITHMS}')
    parser.add_argument('--steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--generations', type=int, default=200,help='The number of generations to run the algorithm.')
    parser.add_argument('--population_size', type=int, default=300,help='The size of the population for the evolutionary algorithm.')
    parser.add_argument('--agents', type=int, default=5,help='The number of agents in the arena.')
    parser.add_argument('--blocks', type=int, default=20,help='The number of blocks in the arena.')
    parser.add_argument('--seed', type=int, default=0,help='The seed for the random number generator.')
    args = parser.parse_args()
    
    if args.evo not in EVOLUTIONARY_ALGORITHMS:
        raise ValueError(f"Script must be one of {EVOLUTIONARY_ALGORITHMS}")
    if args.steps <= 0:
        raise ValueError("Number of steps must be greater than 0")
    if args.generations <= 0:
        raise ValueError("Number of generations must be greater than 0")
    if args.population_size <= 0:
        raise ValueError("Population size must be greater than 0")
    if args.agents <= 0:
        raise ValueError("Number of agents must be greater than 0")
    if args.blocks <= 0:
        raise ValueError("Number of blocks must be greater than 0")
    if args.seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")
    
    main(args.name, 
        args.evo, 
        args.steps,
        args.generations, 
        args.population_size,
        args.agents, 
        args.blocks, 
        args.seed)