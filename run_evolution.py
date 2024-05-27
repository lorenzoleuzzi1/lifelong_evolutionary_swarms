import environment 
import neural_controller 
from utils import neat_sigmoid, plot_data, eaSimpleWithElitism, eaEvoStick 
import random
import numpy as np
import neat
import pickle
from deap import base, creator, tools, algorithms, cma
import argparse
import time
import json
import os
from test_genome import run_episode
import imageio

AVAILABLE_SCRIPTS = ["ga", "cma-es", "neat", "evostick"]
AVAILABLE_INITIAL_SETTINGS = ["easy", "medium", "hard"]

# TODO: make it prettier, maybe classes for each algorithm or Evolution class
def run_neat(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed):
    env = environment.SwarmForagingEnv(objective = [(environment.RED, environment.UP)],
                    size = environment.SIMULATION_ARENA_SIZE,
                    n_agents = n_agents,
                    n_blocks = n_blocks, 
                    n_neighbors = 3,
                    sensor_range = environment.SIMULATION_SENSOR_RANGE,
                    max_wheel_velocity = environment.SIMULATION_MAX_WHEEL_VELOCITY,
                    sensitivity = 0.5,
                    initial_setting = initial_setting,
                    seed = seed)
    env.reset()
    env.render()

    def calculate_fitnesses_neat(genomes, config, n_steps = n_steps):

        for genome_id, genome in genomes:
            genome.fitness = 0.0
            
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            obs, _ = env.reset()

            for step in range(n_steps):
                nn_inputs = env.process_observation(obs)
                nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])
                actions = (2 * nn_outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity

                obs, reward, done, _, _ = env.step(actions)
                genome.fitness += reward
                
                if done:
                    genome.fitness += (n_steps - step) / 2
                    break
            
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    config.genome_config.add_activation('neat_sigmoid', neat_sigmoid)
    config.pop_size = population_size
    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    start = time.time()
    # Run NEAT
    best_individual = p.run(calculate_fitnesses_neat, n_generations)
    end = time.time()
    
    total_reward, frames, info = run_episode(env, best_individual, "neat", n_steps, verbose = False)

    os.mkdir(f"results/{filename}") # Create directory for results

    print(f"Total reward: {total_reward}")
    imageio.mimsave(f'results/{filename}/best_episode.gif', frames, fps=60) # Save the frames as a GIF

    # Plot stats
    bests = stats.get_fitness_stat(np.max)
    avgs = stats.get_fitness_mean()
    medians = stats.get_fitness_median()
    stds = stats.get_fitness_stdev()
    plot_data(bests, avgs = avgs, medians = medians, 
                    completion_fitness=env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP),
                    filename = f"results/{filename}/evolution_plot.png")
    
    # Create a dictonary with stats
    logbook = {
        "best": bests,
        "avg": avgs,
        "median": medians,
        "std": stds
    }
    # Save the logbook as json
    with open(f"results/{filename}/logbook.json", "w") as f:
        json.dump(logbook, f, indent=4)
    # Save winner as pickle
    with open(f"results/{filename}/best_genome.pkl", "wb") as f:
        pickle.dump(best_individual, f)
    # Save the environment as pickle
    with open(f"results/{filename}/env.pkl", "wb") as f:
        pickle.dump(env, f)
    # Save parameters as txt
    with open(f"results/{filename}/parameters.txt", "w") as f:
        f.write(f"Name: {filename}\n")
        f.write(f"Algortihm: NEAT \n")
        f.write(f"Generations: {n_generations}\n")
        f.write(f"Steps: {n_steps}\n")
        f.write(f"Population size: {population_size}\n")
        f.write(f"Number of agents: {n_agents}\n")
        f.write(f"Number of blocks: {n_blocks}\n")
        f.write(f"Initial setting: {initial_setting if initial_setting is not None else 'Random'}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Best fitness: {best_individual.fitness}\n")
        f.write(f"Info: {info}\n")
        f.write(f"Time: {end - start}\n")

def run_ga(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed):
    env = environment.SwarmForagingEnv(objective = [(environment.RED, environment.UP)],
                    size = environment.SIMULATION_ARENA_SIZE,
                    n_agents = n_agents,
                    n_blocks = n_blocks, 
                    n_neighbors = 3,
                    sensor_range = environment.SIMULATION_SENSOR_RANGE,
                    max_wheel_velocity = environment.SIMULATION_MAX_WHEEL_VELOCITY,
                    sensitivity = 0.5,
                    initial_setting = initial_setting,
                    seed = seed)
    env.reset()
    env.render()
    
    input_dim = (env.n_types + 2 + 1) * env.n_neighbors + 2 + env.n_types - 2
    output_dim = 3
    hidden_units = [16]
    layer_sizes = [input_dim] + hidden_units + [output_dim]

    nn = neural_controller.NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")

    def calculate_fitness(individual, n_steps=n_steps):
        fitness = 0
        obs, _ = env.reset()
        
        # Set the weights of the network
        nn.set_weights_from_vector(individual)

        for step in range(n_steps):
            nn_inputs = env.process_observation(obs)
            nn_outputs = np.array(nn.predict(nn_inputs))
            actions = (2 * nn_outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity
                
            obs, reward, done, _, _ = env.step(actions)

            fitness += reward

            if done:
                fitness += (n_steps - step) / 2
                break
        
        return [float(fitness)]
    
    n_elite = int(0.05 * population_size) # 5% of the population will be copied to the next generation (elitism)
    
    # Set up the fitness and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", calculate_fitness)  # Evaluation function

    toolbox.register("attr_float", random.uniform, -5.0, 5.0)  # Attribute generator
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_float, n=nn.total_weights)  # Individual generator
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selTournament, tournsize=int(population_size*0.03))  # Selection function
    # toolbox.register("select", selElitistAndTournament)  # Selection function
    toolbox.register("mate", tools.cxTwoPoint)  # Crossover function
    # toolbox.register("mate", tools.cxOnePoint)  # Crossover function
    # toolbox.register("mate", tools.cxUniform, indpb=0.5)  # 50% chance for each weight
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # Mutation function

    # Statistics to keep track of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)
    stats.register("avg", np.mean)
    stats.register("median", np.median)
    stats.register("std", np.std)
    stats.register("worst", np.min)

    pop = toolbox.population(n=population_size)  # Create a population 
    hof = tools.HallOfFame(n_elite)  # Hall of fame to store the best individual

    start = time.time()
    # Run the genetic algorithm
    pop, log = eaSimpleWithElitism(pop, toolbox, cxpb=0.8, mutpb=0.5, ngen=n_generations,
                                     stats=stats, halloffame=hof, verbose=True)
    end = time.time()
    # mu = 100
    # lambda_ = 200
    # pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=0.8, mutpb=0.5, ngen=n_generations,
    #                                     stats=stats, halloffame=hof, verbose=True)
    
    best_individual = hof[0]
    total_reward, frames, info = run_episode(env, best_individual, "ga", n_steps, verbose = False)
    
    os.mkdir(f"results/{filename}") # Create directory for results

    print(f"Total reward: {total_reward}")
    imageio.mimsave(f'results/{filename}/best_episode.gif', frames, fps=60) # Save the frames as a GIF

    # Plot stats
    bests = []
    avgs = []
    stds = []
    medians = []
    for stat in log:
        bests.append(stat['best'])
        avgs.append(stat['avg'])
        stds.append(stat['std'])
        medians.append(stat['median'])

    plot_data(bests, avgs = avgs, medians = medians, 
                    completion_fitness=env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP),
                    filename = f"results/{filename}/evolution_plot.png")
    
    # Create a dictonary with stats
    logbook = {
        "best": bests,
        "avg": avgs,
        "median": medians,
        "std": stds
    }
    # Save the logbook as json
    with open(f"results/{filename}/logbook.json", "w") as f:
        json.dump(logbook, f, indent=4)
    # Save winner as pickle
    with open(f"results/{filename}/best_genome.pkl", "wb") as f:
        pickle.dump(best_individual, f)
    # Save the environment as pickle
    with open(f"results/{filename}/env.pkl", "wb") as f:
        pickle.dump(env, f)
    # Save parameters as txt
    with open(f"results/{filename}/parameters.txt", "w") as f:
        f.write(f"Name: {filename}\n")
        f.write(f"Algortihm: GA DEAP \n")
        f.write(f"Generations: {n_generations}\n")
        f.write(f"Steps: {n_steps}\n")
        f.write(f"Population size: {population_size}\n")
        f.write(f"Number of agents: {n_agents}\n")
        f.write(f"Number of blocks: {n_blocks}\n")
        f.write(f"Initial setting: {initial_setting if initial_setting is not None else 'Random'}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Best fitness: {np.max(bests)}\n")
        f.write(f"Info: {info}\n")
        f.write(f"Time: {end - start}\n")


def run_cmaes(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed):
    env = environment.SwarmForagingEnv(objective = [(environment.RED, environment.UP)],
                    size = environment.SIMULATION_ARENA_SIZE,
                    n_agents = n_agents,
                    n_blocks = n_blocks, 
                    n_neighbors = 3,
                    sensor_range = environment.SIMULATION_SENSOR_RANGE,
                    max_wheel_velocity = environment.SIMULATION_MAX_WHEEL_VELOCITY,
                    sensitivity = 0.5,
                    initial_setting = initial_setting,
                    seed = seed)
    env.reset()
    env.render()
    
    input_dim = (env.n_types + 2 + 1) * env.n_neighbors + 2 + env.n_types - 2
    output_dim = 3
    hidden_units = [16]
    layer_sizes = [input_dim] + hidden_units + [output_dim]

    nn = neural_controller.NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")

    def calculate_fitness(individual, n_steps=n_steps):
        fitness = 0
        obs, _ = env.reset()
        
        # Set the weights of the network
        nn.set_weights_from_vector(individual)

        for step in range(n_steps):
            nn_inputs = env.process_observation(obs)
            nn_outputs = np.array(nn.predict(nn_inputs))
            actions = (2 * nn_outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity
                
            obs, reward, done, _, _ = env.step(actions)

            fitness += reward
            
            if done:
                fitness += (n_steps - step) / 2
                break
        
        return [float(fitness)]

    # Set up the fitness and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", calculate_fitness)

    # Strategy parameters for CMA-ES
    strategy = cma.Strategy(centroid=[0.0]*nn.total_weights, sigma=5.0, lambda_=population_size)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Statistics to keep track of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)
    stats.register("avg", np.mean)
    stats.register("median", np.median)
    stats.register("std", np.std)
    stats.register("worst", np.min)

    # Using the best strategy to retrieve the best individual
    hof = tools.HallOfFame(1)
    
    start = time.time()
    log = algorithms.eaGenerateUpdate(toolbox, ngen=n_generations, stats=stats, halloffame=hof)
    end = time.time()

    best_individual = hof[0]
    total_reward, frames, info = run_episode(env, best_individual, "cma-es", n_steps, verbose = False)

    os.mkdir(f"results/{filename}") # Create directory for results

    print(f"Total reward: {total_reward}")
    imageio.mimsave(f'results/{filename}/best_episode.gif', frames, fps=60) # Save the frames as a GIF

    # Plot stats
    bests = []
    avgs = []
    stds = []
    medians = []
    for stat in log[1]:
        bests.append(stat['best'])
        avgs.append(stat['avg'])
        stds.append(stat['std'])
        medians.append(stat['median'])

    plot_data(bests, avgs = avgs, medians = medians, 
                    completion_fitness=env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP),
                    filename = f"results/{filename}/evolution_plot.png")
    
    # Create a dictonary with stats
    logbook = {
        "best": bests,
        "avg": avgs,
        "median": medians,
        "std": stds
    }
    # Save the logbook as json
    with open(f"results/{filename}/logbook.json", "w") as f:
        json.dump(logbook, f, indent=4)
    # Save winner as pickle
    with open(f"results/{filename}/best_genome.pkl", "wb") as f:
        pickle.dump(best_individual, f)
    # Save the environment as pickle
    with open(f"results/{filename}/env.pkl", "wb") as f:
        pickle.dump(env, f)
    # Save parameters as txt
    with open(f"results/{filename}/parameters.txt", "w") as f:
        f.write(f"Name: {filename}\n")
        f.write(f"Algortihm: CMA-ES DEAP \n")
        f.write(f"Generations: {n_generations}\n")
        f.write(f"Steps: {n_steps}\n")
        f.write(f"Population size: {population_size}\n")
        f.write(f"Number of agents: {n_agents}\n")
        f.write(f"Number of blocks: {n_blocks}\n")
        f.write(f"Initial setting: {initial_setting if initial_setting is not None else 'Random'}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Best fitness: {np.max(bests)}\n")
        f.write(f"Info: {info}\n")
        f.write(f"Time: {end - start}\n")

def run_evostick(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed):
    env = environment.SwarmForagingEnv(objective = [(environment.RED, environment.UP)],
                    size = environment.SIMULATION_ARENA_SIZE,
                    n_agents = n_agents,
                    n_blocks = n_blocks, 
                    n_neighbors = 3,
                    sensor_range = environment.SIMULATION_SENSOR_RANGE,
                    max_wheel_velocity = environment.SIMULATION_MAX_WHEEL_VELOCITY,
                    sensitivity = 0.5,
                    initial_setting = initial_setting,
                    seed = seed)
    env.reset()
    env.render()
    
    input_dim = (env.n_types + 2 + 1) * env.n_neighbors + 2 + env.n_types - 2
    output_dim = 3
    hidden_units = [16]
    layer_sizes = [input_dim] + hidden_units + [output_dim]

    nn = neural_controller.NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")

    def calculate_fitness(individual, n_steps=n_steps):
        fitness = 0
        obs, _ = env.reset()
        
        # Set the weights of the network
        nn.set_weights_from_vector(individual)

        for step in range(n_steps):
            nn_inputs = env.process_observation(obs)
            nn_outputs = np.array(nn.predict(nn_inputs))
            actions = (2 * nn_outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity
                
            obs, reward, done, _, _ = env.step(actions)

            fitness += reward
            
            if done:
                fitness += (n_steps - step) / 2
                break
        
        return [float(fitness)]
    
    n_elite = int(0.05 * population_size) # 5% of the population will be copied to the next generation (elitism)

    # Set up the fitness and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", calculate_fitness)  # Evaluation function

    toolbox.register("attr_float", random.uniform, -1.0, 1.0)  # Attribute generator
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_float, n=nn.total_weights)  # Individual generator
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", calculate_fitness)
    toolbox.register("select", tools.selBest, k=n_elite)  # Select top 20 individuals
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=1.0)
    toolbox.register("map", map)

    # Statistics to keep track of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)
    stats.register("avg", np.mean)
    stats.register("median", np.median)
    stats.register("std", np.std)
    stats.register("worst", np.min)

    pop = toolbox.population(n=population_size)  # Create a population 
    hof = tools.HallOfFame(2)  # Hall of fame to store the best individual

    start = time.time()
    # Run the genetic algorithm
    pop, log = eaEvoStick(pop, toolbox, n_generations, stats=stats, halloffame=hof, verbose=True)
    end = time.time()

    best_individual = hof[0]
    total_reward, frames, info = run_episode(env, best_individual, "evostick", n_steps, verbose = False)
    os.mkdir(f"results/{filename}") # Create directory for results

    print(f"Total reward: {total_reward}")
    imageio.mimsave(f'results/{filename}/best_episode.gif', frames, fps=60) # Save the frames as a GIF

    # Plot stats
    bests = []
    avgs = []
    stds = []
    medians = []
    for stat in log:
        bests.append(stat['best'])
        avgs.append(stat['avg'])
        stds.append(stat['std'])
        medians.append(stat['median'])

    plot_data(bests, avgs = avgs, medians = medians, 
                    completion_fitness=env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP),
                    filename = f"results/{filename}/evolution_plot.png")
    
    # Create a dictonary with stats
    logbook = {
        "best": bests,
        "avg": avgs,
        "median": medians,
        "std": stds
    }
    # Save the logbook as json
    with open(f"results/{filename}/logbook.json", "w") as f:
        json.dump(logbook, f, indent=4)
    # Save winner as pickle
    with open(f"results/{filename}/best_genome.pkl", "wb") as f:
        pickle.dump(best_individual, f)
    # Save the environment as pickle
    with open(f"results/{filename}/env.pkl", "wb") as f:
        pickle.dump(env, f)
    # Save parameters as txt
    with open(f"results/{filename}/parameters.txt", "w") as f:
        f.write(f"Name: {filename}\n")
        f.write(f"Algortihm: EvoStick DEAP \n")
        f.write(f"Generations: {n_generations}\n")
        f.write(f"Steps: {n_steps}\n")
        f.write(f"Population size: {population_size}\n")
        f.write(f"Number of agents: {n_agents}\n")
        f.write(f"Number of blocks: {n_blocks}\n")
        f.write(f"Initial setting: {initial_setting if initial_setting is not None else 'Random'}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Best fitness: {np.max(bests)}\n")
        f.write(f"Info: {info}\n")
        f.write(f"Time: {end - start}\n")

def main(name, 
        script, 
        n_steps,
        n_generations,
        population_size,
        n_agents, 
        n_blocks,
        initial_setting_type,
        seed):
    
    initial_setting = None
    
    if initial_setting_type == "easy":
        initial_setting = environment.EASY_INITIAL_SETTING
        n_agents = len(initial_setting["agents"])
        n_blocks = len(initial_setting["blocks"]) 
    elif initial_setting_type == "medium":
        initial_setting = environment.MEDIUM_INITIAL_SETTING
        n_agents = len(initial_setting["agents"])
        n_blocks = len(initial_setting["blocks"])
    elif initial_setting_type == "hard":
        initial_setting = environment.HARD_INITIAL_SETTING
        n_agents = len(initial_setting["agents"])
        n_blocks = len(initial_setting["blocks"])
    # else custom using n_agents and n_blocks
    
    filename = f"{name}_{script}_{n_steps}_{n_generations}_{population_size}_{n_agents}_{n_blocks}_{seed}"
    print(f"Running {script} with with the following parameters:")
    print(f"Name: {name}")
    print(f"Number of steps: {n_steps} a step is {environment.TIME_STEP} seconds.")
    print(f"Number of generations: {n_generations}")
    print(f"Population size: {population_size}")
    print(f"Number of agents: {n_agents}")
    print(f"Number of blocks: {n_blocks}")
    print(f"Initial setting: {initial_setting if initial_setting is not None else 'Random'}")
    print(f"Seed: {seed}")

    if script == "neat":
        run_neat(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed)
    elif script == "ga":
        run_ga(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed)
    elif script == "cma-es":
        run_cmaes(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed)
    elif script == "evostick":
        run_evostick(filename, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input for evolutionary swarm')
    parser.add_argument('--name', type=str, default="test", help=f'The name of the experiment.')
    parser.add_argument('--script', type=str, default="neat", help=f'The name of the script to run. Must be one of: {AVAILABLE_SCRIPTS}')
    parser.add_argument('--n_steps', type=int, default=500, help='The number of steps of each episode.')
    parser.add_argument('--n_generations', type=int, default=200,help='The number of generations to run the algorithm.')
    parser.add_argument('--population_size', type=int, default=300,help='The size of the population for the evolutionary algorithm.')
    parser.add_argument('--n_agents', type=int, default=5,help='The number of agents in the arena.')
    parser.add_argument('--n_blocks', type=int, default=20,help='The number of blocks in the arena.')
    parser.add_argument('--initial_setting', type=str, default=None,help=f'The predifined initial setting of the environment. Must be one of: {AVAILABLE_INITIAL_SETTINGS}')
    parser.add_argument('--seed', type=int, default=0,help='The seed for the random number generator.')
    args = parser.parse_args()
    
    if args.script not in AVAILABLE_SCRIPTS:
        raise ValueError(f"Script must be one of {AVAILABLE_SCRIPTS}")
    if args.initial_setting not in AVAILABLE_INITIAL_SETTINGS and args.initial_setting is not None:
        raise ValueError(f"Initial setting must be one of {AVAILABLE_INITIAL_SETTINGS}")
    if args.n_steps <= 0:
        raise ValueError("Number of steps must be greater than 0")
    if args.n_generations <= 0:
        raise ValueError("Number of generations must be greater than 0")
    if args.population_size <= 0:
        raise ValueError("Population size must be greater than 0")
    if args.n_agents <= 0:
        raise ValueError("Number of agents must be greater than 0")
    if args.n_blocks <= 0:
        raise ValueError("Number of blocks must be greater than 0")
    if args.seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")
    
    main(args.name, 
        args.script, 
        args.n_steps,
        args.n_generations, 
        args.population_size,
        args.n_agents, 
        args.n_blocks, 
        args.initial_setting,
        args.seed)