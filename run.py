import environment 
import neural_controller
import utils
import random
import numpy as np
import neat
import pickle
from deap import base, creator, tools, algorithms, cma

# TODO: make it prettier

def run_neat(initial_settings, num_generations, filename):
    env = environment.Environment(objective = [(environment.RED, environment.UP)],
                   size = environment.SIMULATION_ARENA_SIZE, 
                   n_agents = len(initial_settings['agents']), 
                   n_blocks = len(initial_settings['blocks']),
                   n_neighbors = 3,
                   sensor_range = environment.SIMULATION_SENSOR_RANGE,
                   sensor_angle = 360,
                   max_wheel_velocity = environment.SIMULATION_MAX_WHEEL_VELOCITY,
                   sensitivity = 0.2,
                   initial_setting = initial_settings)
    
    env.reset()

    def calculate_fitnesses_neat(genomes, config, n_steps = 200, verbose=False):
        flag_done = False
        best_steps = n_steps

        for genome_id, genome in genomes:
            genome.fitness = 0.0
            
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            obs, _ = env.reset()
            if verbose: env.print_env()

            for step in range(n_steps):
                nn_inputs = env.process_observation(obs)
                nn_outputs = [net.activate(nn_input) for nn_input in nn_inputs]
                actions = np.dot(nn_outputs, env.max_wheel_velocity)

                obs, reward, done, _, _ = env.step(actions)
                genome.fitness += reward

                if verbose:
                    # print("NN inputs: ", nn_inputs)
                    print("Action: ", actions)
                    print("\nStep", step)
                    env.print_env()
                    print("Observation: ", obs)
                    print("Reward: ", reward)
                
                if done:
                    genome.fitness += (n_steps - step) / 2
                    flag_done = True
                    if best_steps > step:
                        best_steps = step
                    break
            
        if flag_done:
            print(f"Done in {best_steps} steps")
    
    # Set configuration file
    config_path = "./neat_config.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(calculate_fitnesses_neat, num_generations)

    # Plot stats
    bests = stats.get_fitness_stat(np.max)
    avgs = stats.get_fitness_mean()
    medians = stats.get_fitness_median()
    stds = stats.get_fitness_stdev()
    utils.plot_data(bests, avgs = avgs, medians = medians, 
                    completion_fitness=env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP),
                    filename = f"results/plots/{filename}_plot.png")
    
    # Save winner as pickle
    with open(f"results/winners/{filename}_best.pkl", "wb") as f:
        pickle.dump(winner, f)
  
def run_ga(initial_settings, num_generations, filename):
    env = environment.Environment(objective = [(environment.RED, environment.UP)],
                   size = environment.SIMULATION_ARENA_SIZE, 
                   n_agents = len(initial_settings['agents']), 
                   n_blocks = len(initial_settings['blocks']),
                   n_neighbors = 3,
                   sensor_range = environment.SIMULATION_SENSOR_RANGE,
                   sensor_angle = 360,
                   max_distance_covered_per_step = environment.SIMULATION_MAX_DISTANCE,
                   sensitivity = 0.2,
                   initial_setting = initial_settings)
    
    input_dim = (env.n_types + 2) * env.n_neighbors + env.n_types - 2
    output_dim = 2
    hidden_units = [16]
    layer_sizes = [input_dim] + hidden_units + [output_dim]

    nn = neural_controller.NeuralController(layer_sizes, hidden_activation="sigmoid", output_activation="sigmoid")

    def calculate_fitness(individual, n_steps=200, verbose = False):
        fitness = 0
        obs, _ = env.reset()
        
        # Set the weights of the network
        nn.set_weights_from_vector(individual)

        for step in range(n_steps):
            nn_inputs = env.process_observation(obs)
            nn_outputs = np.array(nn.predict(nn_inputs))
            actions = nn_outputs * np.array([env.max_distance_covered_per_step, env.sensor_angle])
                
            obs, reward, done, _, _ = env.step(actions)

            fitness += reward

            if verbose:
                print("\nStep", step)
                # print("Observation: ", obs)
                # print("NN inputs: ", nn_inputs)
                print("Action: ", actions)
                env.print_env()
                print("Reward: ", reward)
            if done:
                fitness += (n_steps - step) / 2
                break
        
        return [float(fitness)]

    pop_size = 200
    num_elite = int(0.05 * pop_size) # 5% of the population will be copied to the next generation (elitism)
    num_removed = int(0.1 * pop_size) # 10% of the population will be removed and randomly replaced
    tournament_size = int(0.03 * pop_size)
    offspring_size = pop_size - num_elite - num_removed

    # Set up the fitness and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("evaluate", calculate_fitness)  # Evaluation function

    toolbox.register("attr_float", random.uniform, -5.0, 5.0)  # Attribute generator
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_float, n=nn.total_weights)  # Individual generator
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register("select", tools.selTournament, tournsize=3)  # Selection function
    toolbox.register("select", utils.selElitistAndTournament)  # Selection function
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

    pop = toolbox.population(n=pop_size)  # Create a population 
    hof = tools.HallOfFame(1)  # Hall of fame to store the best individual

    # Run the genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3, ngen=num_generations,
                                    stats=stats, halloffame=hof, verbose=True)
    
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

    utils.plot_data(bests, avgs = avgs, medians = medians, 
                    completion_fitness=env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP),
                    filename = f"results/plots/{filename}_plot.png")
    
    # Save weights of the best individual
    best_individual = hof[0]
    np.save(f"results/winners/{filename}_best.npy", best_individual)

def run_cmaes(initial_settings, num_generations, filename):
    env = environment.Environment(objective = [(environment.RED, environment.UP)],
                   size = environment.SIMULATION_ARENA_SIZE, 
                   n_agents = len(initial_settings['agents']), 
                   n_blocks = len(initial_settings['blocks']),
                   n_neighbors = 3,
                   sensor_range = environment.SIMULATION_SENSOR_RANGE,
                   sensor_angle = 360,
                   max_distance_covered_per_step = environment.SIMULATION_MAX_DISTANCE,
                   sensitivity = 0.2,
                   initial_setting = initial_settings)
    env.reset()
    input_dim = (env.n_types + 2) * env.n_neighbors + env.n_types - 2
    output_dim = 2
    hidden_units = [16]
    layer_sizes = [input_dim] + hidden_units + [output_dim]

    nn = neural_controller.NeuralController(layer_sizes, hidden_activation="sigmoid", output_activation="sigmoid")

    def calculate_fitness(individual, n_steps=200, verbose = False):
        fitness = 0
        obs, _ = env.reset()
        
        # Set the weights of the network
        nn.set_weights_from_vector(individual)

        for step in range(n_steps):
            nn_inputs = env.process_observation(obs)
            nn_outputs = np.array(nn.predict(nn_inputs))
            actions = nn_outputs * np.array([env.max_distance_covered_per_step, env.sensor_angle])
                
            obs, reward, done, _, _ = env.step(actions)

            fitness += reward

            if verbose:
                print("\nStep", step)
                # print("Observation: ", obs)
                # print("NN inputs: ", nn_inputs)
                print("Action: ", actions)
                env.print_env()
                print("Reward: ", reward)
            
            if done:
                fitness += (n_steps - step) / 2
                break
        
        return [float(fitness)]
    
    population_size = 200

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

    log = algorithms.eaGenerateUpdate(toolbox, ngen=num_generations, stats=stats, halloffame=hof)

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

    utils.plot_data(bests, avgs = avgs, medians = medians, 
                    completion_fitness=env.n_task * (environment.REWARD_PICK + environment.REWARD_DROP),
                    filename = f"results/plots/{filename}_plot.png")
    
    # Save weights of the best individual
    best_individual = hof[0]
    np.save(f"results/winners/{filename}_best.npy", best_individual)
