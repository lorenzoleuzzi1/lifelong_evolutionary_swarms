import environment
import neural_controller 
from utils import neat_sigmoid, plot_evolution, eaSimpleWithElitism, eaEvoStick 
import random
import numpy as np
import neat
import pickle
from deap import base, creator, tools, algorithms, cma
import time
import json
import os
import imageio
import copy

EVOLUTIONARY_ALGORITHMS = ['neat', 'ga', 'cma-es', 'evostick']
DEAP_ALGORITHMS = ['cma-es', 'ga', 'evostick']

# TODO: finish it and check if it is correct
# TODO put together results and plot them
class EvoSwarmExperiment:

    def __init__(self,
                evolutionary_algorithm : str = None,
                name : str = None,
                population_size : int = None,
                controller_deap : neural_controller.NeuralController = None,
                env : environment.SwarmForagingEnv = None,
                config_path_neat : str = None,
                seed : int = None
                ):
        
        self.evolutionary_algorithm = evolutionary_algorithm
        self.name = name
        self.population_size = population_size
        self.env = env
        self.controller_deap = controller_deap
        self.config_path_neat = config_path_neat
        self.seed = seed

        self.experiment_name = None
        self.best_individual = None
        self.time_elapsed = None
        self.toolbox_deap = None
        self.population = None
        if env is not None:
            self.target_color = env.target_color
        else:
            self.target_color = None
        self.prev_target_color = None

        self.n_eval_forgetting = None
        self.fitness_forgetting = []
    
    def load(self, folder_path):
        # self.experiment_name = folder_path.split("/")[-1]
        # Load env 
        with open(folder_path + '/env.pkl', 'rb') as f:
            self.env = pickle.load(f)
        # Load the best genome
        with open(folder_path + '/best_genome.pkl', 'rb') as f:
            self.best_individual = pickle.load(f)
        # Load the logbook
        with open(folder_path + '/logbook.json', 'r') as f:
            self.log = json.load(f)
        # Load the experiment json
        with open(folder_path + '/experiment.json', 'r') as f:
            experiment = json.load(f)
        # Load the population
        with open(folder_path + '/population.pkl', 'rb') as f:
            self.population = pickle.load(f)
        # Set attributes
        self.evolutionary_algorithm = experiment["algorithm"]
        self.steps = experiment["ep_duration"]
        # self.generations = experiment["generations"]
        self.population_size = experiment["population_size"]
        self.experiment_name = experiment["name"]
        self.name = experiment["name"].split("_")[0]
        self.time_elapsed = experiment["time"]
        self.seed = experiment["seed"]
        self.target_color = experiment["target_color"]
        # self.prev_target = experiment["target_color"]
        # Other loads
        if self.evolutionary_algorithm == "neat":
            with open(folder_path + '/neat_config.txt', 'r') as f:
                self.config_path_neat = f.read()
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            with open(folder_path + '/controller.pkl', 'rb') as f:
                self.controller_deap = pickle.load(f)
            with open(folder_path + '/toolbox.pkl', 'rb') as f:
                self.toolbox_deap = pickle.load(f)
        
    def change_objective(self, objective):
        self.fitness_forgetting = []
        self.experiment_name = f"{self.experiment_name}_drift{self.target_color}{objective}"
        self.prev_target_color = self.target_color
        self.target_color = objective
        
        self.env.target_color = objective
     
    def _calculate_fitness_neat(self, genomes, config):
        self.env.target_color = self.target_color
        for genome_id, genome in genomes:
            genome.fitness = 0.0
            
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            obs, _ = self.env.reset(seed=self.seed)
            
            while True:
                nn_inputs = self.env.process_observation(obs)
                nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])
                actions = (2 * nn_outputs - 1) * self.env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity

                obs, reward, done, truncated, _ = self.env.step(actions)
                genome.fitness += reward
                
                if done or truncated:
                    break

        # ----- EVALUATE FORGETTING -----
        if self.n_eval_forgetting is not None and self.prev_target_color is not None:
            self.env.target_color = self.prev_target_color
            # Take top 10 genomes and evaluate them on the previous task
            top_genomes = copy.deepcopy(genomes)
            top_genomes.sort(key=lambda x: x[1].fitness, reverse=True)
            top_genomes = top_genomes[:self.n_eval_forgetting]
            
            for _, top_genome in top_genomes:
                top_genome.fitness = 0.0
                net = neat.nn.FeedForwardNetwork.create(top_genome, config)
                obs, _ = self.env.reset(seed=self.seed)
                
                while True:
                    nn_inputs = self.env.process_observation(obs)
                    nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])
                    actions = (2 * nn_outputs - 1) * self.env.max_wheel_velocity

                    obs, reward, done, truncated, _ = self.env.step(actions)
                    top_genome.fitness += reward

                    if done or truncated:
                        break
            # Calculate the average fitness of the top genomes
            avg_fitness = sum([top_genome.fitness for _, top_genome in top_genomes]) / len(top_genomes)
            print(f"Forgetting: {avg_fitness}")
            self.fitness_forgetting.append(avg_fitness)
        # -------------------------------
    
    def _calculate_fitness_deap(self, individual):
        fitness = 0
        obs, _ = self.env.reset(seed=self.seed)
        
        self.controller_deap.set_weights_from_vector(individual) # Set the weights of the network
        
        while True:
            nn_inputs = self.env.process_observation(obs)
            nn_outputs = np.array(self.controller_deap.predict(nn_inputs))
            actions = (2 * nn_outputs - 1) * self.env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity
                
            obs, reward, done, truncated, _ = self.env.step(actions)

            fitness += reward

            if done or truncated:
                break
        
        return [float(fitness)]
    
    def _run_neat(self, generations):
        if self.config_path_neat is None:
            raise ValueError("Neat config path is not set. Set the path to the config file first.")

        # Set configuration file
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, self.config_path_neat)
        
        config.genome_config.add_activation('neat_sigmoid', neat_sigmoid)
        config.pop_size = self.population_size
        stats = neat.StatisticsReporter()
        if self.population is None:
            self.population = neat.Population(config)
            self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(stats)

        start = time.time()
        # Run NEAT
        self.best_individual = self.population.run(self._calculate_fitness_neat, generations)
        end = time.time()
        self.time_elapsed = end - start
        self.log = stats
    
    def _run_ga(self, generations):
        if self.controller_deap is None:
            raise ValueError("DEAP controller is not set. Set the controller first.")
        
        if self.toolbox_deap is None:
            # Set up the fitness and individual
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
            creator.create("Individual", list, fitness=creator.FitnessMax)
            self.toolbox_deap = base.Toolbox()
            self.toolbox_deap.register("evaluate", self._calculate_fitness_deap)  # Evaluation function
            self.toolbox_deap.register("attr_float", random.uniform, -1.0, 1.0)  # Attribute generator
            self.toolbox_deap.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox_deap.attr_float, n=self.controller_deap.total_weights)  # Individual generator
            self.toolbox_deap.register("population", tools.initRepeat, list, self.toolbox_deap.individual)
            self.toolbox_deap.register("select", tools.selTournament, tournsize=int(self.population_size*0.03))  # Selection function
            # toolbox.register("select", selElitistAndTournament)  # Selection function
            self.toolbox_deap.register("mate", tools.cxTwoPoint)  # Crossover function
            # toolbox.register("mate", tools.cxOnePoint)  # Crossover function
            # toolbox.register("mate", tools.cxUniform, indpb=0.5)  # 50% chance for each weight
            self.toolbox_deap.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # Mutation function

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("best", np.max)
        stats.register("avg", np.mean)
        stats.register("median", np.median)
        stats.register("std", np.std)
        stats.register("worst", np.min)
        
        if self.population is None:
            self.population = self.toolbox_deap.population(n=self.population_size)  # Create a new population 
        
        n_elite = int(0.05 * self.population_size) # 5% of the population will be copied to the next generation (elitism)
        hof = tools.HallOfFame(n_elite)  # Hall of fame to store the best individual

        start = time.time()
        # Run the genetic algorithm
        # TODO use standard algorithm
        self.population, log = algorithms.eaSimple(self.population, self.toolbox_deap, cxpb=0.8, mutpb=0.01,
                                                    ngen=generations, stats=stats, halloffame=hof, verbose=True)
        # self.population, log = eaSimpleWithElitism(self.population, self.toolbox_deap, cxpb=0.8, mutpb=0.1, 
        #                                            ngen=generations, stats=stats, halloffame=hof, verbose=True)
        end = time.time()
        
        self.best_individual = hof[0]
        self.time_elapsed = end - start
        self.log = log

    def _run_cmaes(self, generations):
        if self.controller_deap is None:
            raise ValueError("DEAP controller is not set. Set the controller first.")
        
        if self.toolbox_deap is None:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
            creator.create("Individual", list, fitness=creator.FitnessMax)
            self.toolbox_deap = base.Toolbox()
            self.toolbox_deap.register("evaluate", self._calculate_fitness_deap)
            # Strategy parameters for CMA-ES
            strategy = cma.Strategy(centroid=[0.0]*self.controller_deap.total_weights, sigma=5.0, lambda_=self.population_size)
            self.toolbox_deap.register("generate", strategy.generate, creator.Individual)
            self.toolbox_deap.register("update", strategy.update)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("best", np.max)
        stats.register("avg", np.mean)
        stats.register("median", np.median)
        stats.register("std", np.std)
        stats.register("worst", np.min)

        hof = tools.HallOfFame(2)
        
        start = time.time()
        self.population, log = algorithms.eaGenerateUpdate(self.toolbox_deap, ngen=generations, stats=stats, halloffame=hof)
        end = time.time()
        
        self.best_individual = hof[0]
        self.time_elapsed = end - start
        self.log = log

    def _run_evostick(self, generations):
        if self.controller_deap is None:
            raise ValueError("DEAP controller is not set. Set the controller first.")
        
        n_elite = int(0.2 * self.population_size) # 20% of the population will be copied to the next generation (elitism)
        if self.toolbox_deap is None:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
            creator.create("Individual", list, fitness=creator.FitnessMax)
            self.toolbox_deap = base.Toolbox()
            self.toolbox_deap.register("evaluate", self._calculate_fitness_deap)  # Evaluation function
            self.toolbox_deap.register("attr_float", random.uniform, -1.0, 1.0)  # Attribute generator
            self.toolbox_deap.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox_deap.attr_float, n=self.controller_deap.total_weights)  # Individual generator
            self.toolbox_deap.register("population", tools.initRepeat, list, self.toolbox_deap.individual)
            self.toolbox_deap.register("select", tools.selBest, k=n_elite)  # Select top 20 individuals
            self.toolbox_deap.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=1.0)
            self.toolbox_deap.register("map", map)

        stats = tools.Statistics(lambda ind: ind.fitness.values) # Statistics to keep track of the evolution
        stats.register("best", np.max)
        stats.register("avg", np.mean)
        stats.register("median", np.median)
        stats.register("std", np.std)
        stats.register("worst", np.min)

        if self.population is None:
            self.population = self.toolbox_deap.population(n=self.population_size)  # Create a new population 
        
        hof = tools.HallOfFame(n_elite)  # Hall of fame to store the best individual

        start = time.time()
        # Run the genetic algorithm
        self.population, log = eaEvoStick(self.population, self.toolbox_deap, generations, stats=stats, halloffame=hof, verbose=True)
        end = time.time()

        self.best_individual = hof[0]
        self.time_elapsed = end - start
        self.log = log

    def run_best(self, save = True, verbose = False):
        if self.best_individual is None:
            raise ValueError("Best individual is not set. Set it or tun the evolutionary algorithm first.")
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        
        # Set up the neural network controller
        if self.evolutionary_algorithm == 'neat':
            if self.config_path_neat is None:
                raise ValueError("Neat config path is not set. Set the path to the config file first.")
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation, self.config_path_neat)
            
            config.genome_config.add_activation('neat_sigmoid', neat_sigmoid)
            controller_neat = neat.nn.FeedForwardNetwork.create(self.best_individual, config)
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            if self.controller_deap is None:
                raise ValueError("DEAP controller is not set. Set the controller first.")
            self.controller_deap.set_weights_from_vector(self.best_individual)
        
        frames = []
        done = False
        total_reward = 0
        self.env.target_color = self.target_color
        obs, _ = self.env.reset(seed=self.seed)
        frames.append(self.env.render(verbose))
        
        while True:
            inputs = self.env.process_observation(obs)
            
            if self.evolutionary_algorithm == 'neat':
                outputs = np.array([controller_neat.activate(input) for input in inputs])
            elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
                outputs = np.array(self.controller_deap.predict(inputs))

            actions = (2 * outputs - 1) * self.env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity

            obs, reward, done, truncated, info = self.env.step(actions)
            frames.append(self.env.render(verbose))
            total_reward += reward

            if done or truncated:
                break
        
        if save:
            imageio.mimsave(f"results/{self.experiment_name}/episode.gif", frames, fps = 60)
        
        return total_reward, info
    
    def _save_results(self):
        # Plot stats
        if self.evolutionary_algorithm == "neat":
            
            bests = self.log.get_fitness_stat(np.max)
            avgs = self.log.get_fitness_mean()
            medians = self.log.get_fitness_median()
            stds = self.log.get_fitness_stdev()
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            bests = []
            avgs = []
            stds = []
            medians = []
            for stat in self.log:
                bests.append(stat['best'])
                avgs.append(stat['avg'])
                stds.append(stat['std'])
                medians.append(stat['median'])
        
        plot_evolution(bests, avgs = avgs, medians = medians, 
                       filename = f"results/{self.experiment_name}/evolution_plot.png")
        
        total_reward, info = self.run_best(save = True)
        # Stats
        logbook = {
            "best": bests,
            "avg": avgs,
            "median": medians,
            "std": stds,
            "forgetting": self.fitness_forgetting
        }
        # Experiment info
        experiment = {
            "name": self.experiment_name,
            "algorithm": self.evolutionary_algorithm,
            "generations":len(bests),
            "ep_duration": self.env.duration,
            "population_size": self.population_size,
            "target_color": self.target_color,
            "agents": self.env.n_agents,
            "blocks": self.env.n_blocks,
            "seed": self.seed,
            "best_fitness": total_reward,
            "info": info,
            "retrieved": len(info["retrieved"]),
            "time": self.time_elapsed
        }
        # Save the logbook as json
        with open(f"results/{self.experiment_name}/logbook.json", "w") as f:
            json.dump(logbook, f, indent=4)
        # Save experiment info as json
        with open(f"results/{self.experiment_name}/experiment.json", "w") as f:
            json.dump(experiment, f, indent=4)
        # Save the environment as pickle
        with open(f"results/{self.experiment_name}/env.pkl", "wb") as f:
            pickle.dump(self.env, f)
        # Save winner as pickle
        with open(f"results/{self.experiment_name}/best_genome.pkl", "wb") as f:
            pickle.dump(self.best_individual, f)
        # Save the population as pickle
        with open(f"results/{self.experiment_name}/population.pkl", "wb") as f:
            pickle.dump(self.population, f)
        if self.evolutionary_algorithm == "neat":
            # Save the neat config file
            with open(f"results/{self.experiment_name}/neat_config.txt", "w") as f:
                f.write(str(self.config_path_neat))
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            # Save the deap controller
            with open(f"results/{self.experiment_name}/controller.pkl", "wb") as f:
                pickle.dump(self.controller_deap, f)
            # Save the toolbox
            with open(f"results/{self.experiment_name}/toolbox.pkl", "wb") as f:
                pickle.dump(self.toolbox_deap, f)
        
    def run(self, generations, n_eval_forgetting = None):
        self.n_eval_forgetting = n_eval_forgetting
        
        if self.name is None:
            raise ValueError("Name is not set. Set the name of the experiment first.")
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        if self.evolutionary_algorithm is None:
            raise ValueError("Evolutionary algorithm is not set. Set the evolutionary algorithm first.")
        if self.population_size is None:
            raise ValueError("Population size is not set. Set the population size first.")
        if self.experiment_name is None:
            self.experiment_name = f"{self.name}_{self.evolutionary_algorithm}_{self.env.duration}_{generations}_{self.population_size}_{self.env.n_agents}_{self.env.n_blocks}_{self.seed}"
        os.makedirs(f"results/{self.experiment_name}", exist_ok=True) # Create directory for the experiment results

        print(f"\n{self.experiment_name}")
        print(f"Running {self.evolutionary_algorithm} with with the following parameters:")
        print(f"Name: {self.name}")
        print(f"Duration of episode: {self.env.duration} in steps (one step is {environment.TIME_STEP} seconds)")
        print(f"Number of generations: {generations}")
        print(f"Population size: {self.population_size}")
        print(f"Number of agents: {self.env.n_agents}")
        print(f"Number of blocks: {self.env.n_blocks}")
        print(f"Seed: {self.seed}")
        
        self.env.reset(seed=self.seed)
        self.env.render() # Show the environment

        if self.evolutionary_algorithm == "neat":
            self._run_neat(generations)
        elif self.evolutionary_algorithm == "ga":
            self._run_ga(generations)
        elif self.evolutionary_algorithm == "cma-es":
            self._run_cmaes(generations)
        elif self.evolutionary_algorithm == "evostick":
            self._run_evostick(generations)
        else:
            raise ValueError(f"Evolutionary algorithm must be one of {EVOLUTIONARY_ALGORITHMS}")
        
        self._save_results()
