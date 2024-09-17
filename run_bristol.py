from experiment import LifelongEvoSwarmExperiment, EVOLUTIONARY_ALGORITHMS
from environment import SwarmForagingEnv, BLUE, RED
from neural_controller import NeuralController
import argparse
import numpy as np
import neat
from utils import neat_sigmoid

def main(name, 
        steps,
        n_agents,
        generations,
        population_size,
        seed,
        workers):
    
    size = 16
    sensor_range = 3.5 # should be 4 but 3 to be sure
    sensitivity = 1.4
    max_wheel_speed = 0.4
    
    initial_setting_bristol = {}
    initial_setting_bristol['agents'] =  np.array([
        [1.5, 8.0], # Agent 0
        [1.5, 4.0], # Agent 1
        [1.5, 12.0] # Agent 2
        ])[:n_agents]
    initial_setting_bristol['headings'] = np.array([176 , 181, 178])[:n_agents]
    initial_setting_bristol['blocks'] = np.array([
            [10.8, 2.8], # Carrier 100
            [8.0, 4.0], # Carrier 101
            [11.6, 6.8], # Carrier 102
            [8.0, 8.8],  # Carrier 103
            [12.4, 12.8], # Carrier 104
            [6.4, 14], # Carrier 105
            [10, 10.8] # Carrier 106
        ])
    initial_setting_bristol['colors'] = np.array([3, 5, 3, 4, 4, 5, 3])
    n_colors = len(np.unique(initial_setting_bristol['colors']))
    
    env = SwarmForagingEnv(n_agents = len(initial_setting_bristol['agents']), 
                           n_blocks = len(initial_setting_bristol['blocks']), 
                           target_color = 3, # RED
                           duration=steps, size = size, max_retrieves=3,
                           sensor_range = sensor_range,
                           sensitivity = sensitivity,
                           max_wheel_velocity = max_wheel_speed,
                           n_colors = n_colors,
                           repositioning=False,
                           efficency_reward=True,
                           see_other_agents=False)
    
    config_path_neat = "config-feedforward.txt"
    # Set configuration file
    config_neat = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_neat)
    config_neat.genome_config.add_activation('neat_sigmoid', neat_sigmoid)
    config_neat.pop_size = population_size
    obs_example = env.reset(seed=seed)[0]
    config_neat.genome_config.num_inputs = len(env.process_observation(obs_example)[0])
    config_neat.genome_config.input_keys = [-i - 1 for i in range(config_neat.genome_config.num_inputs)]
    
    experiment = LifelongEvoSwarmExperiment(env = env, name = name, evolutionary_algorithm="neat", 
                                    population_size=population_size, 
                                    config_neat=config_neat, 
                                    env_initial_state=initial_setting_bristol,
                                    seed = seed)

    experiment.run(generations, n_workers = workers)
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evolutionary swarm parameters.')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    parser.add_argument('--steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--agents', type=int, default=3,help='The number of agents in the arena.')
    parser.add_argument('--generations', type=int, default=200,help='The number of generations to run the algorithm.')
    parser.add_argument('--population', type=int, default=300,help='The size of the population for the evolutionary algorithm.')
    parser.add_argument('--seed', type=int, default=0,help='The seed for the random number generator.')
    parser.add_argument('--workers', type=int, default=1, help='The number of workers to run the algorithm.')
    args = parser.parse_args()
    
    if args.steps <= 0:
        raise ValueError("Number of steps must be greater than 0")
    if args.agents != 1 and args.agents != 3:
        raise ValueError("Number of agents must be 1 or 3")
    if args.generations <= 0:
        raise ValueError("Number of generations must be greater than 0")
    if args.population <= 0:
        raise ValueError("Population size must be greater than 0")
    if args.seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")
    if args.workers <= 0:
        raise ValueError("Number of workers must be greater than 0")
    
    main(args.name, 
        args.steps,
        args.agents,
        args.generations, 
        args.population,
        args.seed,
        args.workers
        )