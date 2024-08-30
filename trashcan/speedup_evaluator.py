import numpy as np
import neat
import environment
import utils
from trashcan.parallel import ThreadedEvaluator, ParallelEvaluator
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

time_create_env = 0.0
for i in range(2*100):
    start = time.time()
    env = environment.SwarmForagingEnv(target_color= environment.RED, n_agents = 5, n_blocks = 30, 
                                    distribution="uniform", duration=800) 
    time_create_env += time.time() - start
print(f"Time to create environment: {time_create_env}")

seed = 3

def eval_genomes(genomes, config):
    env = environment.SwarmForagingEnv(target_color= environment.RED, n_agents = 5, n_blocks = 30, 
                                   distribution="uniform", duration=800) 

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs, _ = env.reset(seed=seed)
        
        while True:
            nn_inputs = env.process_observation(obs)
            nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])
            actions = (2 * nn_outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity

            obs, reward, done, truncated, _ = env.step(actions)
            genome.fitness += reward

            if done or truncated:
                break

def eval_genome(genome, config):
    env = environment.SwarmForagingEnv(target_color= environment.RED, n_agents = 5, n_blocks = 30, 
                                   distribution="uniform", duration=800) 
    fitness = 0.0

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    obs, _ = env.reset(seed=seed)

    while True:
        nn_inputs = env.process_observation(obs)
        nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])
        actions = (2 * nn_outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity

        obs, reward, done, truncated, _ = env.step(actions)
        fitness += reward

        if done or truncated:
            break

    return fitness


config_path = "config-feedforward.txt"
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
config.genome_config.add_activation('neat_sigmoid', utils.neat_sigmoid)
config.pop_size = 100
# Create core evolution algorithm class
p = neat.Population(config)

# Add reporter for fancy statistical result
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Sequential evaluation
start_seq = time.time()
# winner = p.run(eval_genomes, 2)
print("Sequential evaluation took: ", time.time() - start_seq)

# Parallel evaluation
evaluator_t = ThreadedEvaluator(multiprocessing.cpu_count(), eval_genome)
start_par = time.time()
# winner = p.run(evaluator_t.evaluate, 2)
evaluator_t.stop()
print(f"Parallel with {multiprocessing.cpu_count()} workers evaluation took: ", time.time() - start_par)
print(f"Time to create environment: {time_create_env}")
# evaluator_p = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
# start_par = time.time()
# winner = p.run(evaluator_p.evaluate, 2)
# print("Parallel evaluation took: ", time.time() - start_par)
