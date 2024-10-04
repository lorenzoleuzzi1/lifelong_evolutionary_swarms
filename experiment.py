import neat.config
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
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import gymnasium as gym

EVOLUTIONARY_ALGORITHMS = ['neat', 'ga', 'cma-es', 'evostick']
DEAP_ALGORITHMS = ['cma-es', 'ga', 'evostick']

FREQUENCY_EVAL_RETENTION = 10

# TODO: dont save all the info for deap, just neat handles drifts
# TODO: create two separate classes and use checkpoints
# TODO: base class, DEAP class, NEAT class
# TODO: only NEAT handles drifts
class LifelongEvoSwarmExperiment:

    def __init__(self,
                evolutionary_algorithm : str = None,
                name : str = None,
                population_size : int = None,
                controller_deap : neural_controller.NeuralController = None,
                env : environment.SwarmForagingEnv = None,
                env_initial_state : dict = None,
                config_neat : neat.config.Config = None,
                reg_lambdas : dict = {"gd": 6.0, "wp": [0.4, 0.3]}, 
                seed : int = None,
                n_env : int = 1, # Number of environments for evaluating fitness
                n_workers : int = 1
                ):
        
        self.evolutionary_algorithm = evolutionary_algorithm
        self.name = name
        self.population_size = population_size
        self.env = env
        self.env_initial_state = env_initial_state
        self.controller_deap = controller_deap
        self.config_neat = config_neat
        self.reg_lambdas = reg_lambdas
        self.seed = seed
        self._current_generation = 0

        self.experiment_name = None
        self.best_individual = None
        self.time_elapsed = None
        self.population = None
        if env is not None:
            self.target_color = env.target_color
        else:
            self.target_color = None
        self.prev_target_color = None

        # they are set when calling run method TODO: maybe init?? n_workers yes but the other no bc the change between runs
        self.n_workers = n_workers
        self.n_env = n_env

        self.eval_retention = None
        self.regularization_retention = None
        self.retention_pop_fitness = []
        self.retention_top_fitness = []
        self.fitness_no_penalty = []
    
    def load(self, folder_path):
        # Print files in the path
        print(f"Loading experiment from {folder_path}")
        print(os.listdir(folder_path))
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
        self.prev_target_color = experiment["prev_target_color"]
        # self.prev_target = experiment["target_color"]
        # Other loads
        if self.evolutionary_algorithm == "neat":
            with open(folder_path + '/neat_config.pkl', 'rb') as f:
                self.config_neat = pickle.load(f)
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            with open(folder_path + '/controller.pkl', 'rb') as f:
                self.controller_deap = pickle.load(f)
    
    def drift(self, new_target, env_initial_state = None):
        if self.evolutionary_algorithm != "neat":
            raise ValueError(f"Drifts are implemented only for NEAT. Got {self.evolutionary_algorithm}.")
        
        self.retention_pop_fitness = []
        self.retention_top_fitness = []
        self.fitness_no_penalty = []
        self.experiment_name = f"{self.experiment_name}_drift{self.target_color}{new_target}"
        self.prev_target_color = self.target_color
        self.target_color = new_target
        self._current_generation = 0
        
        self.env.target_color = new_target

        if env_initial_state is not None:
            self.env_initial_state = env_initial_state
    
    def _run_episode(self, genome, config, env, n_env = 10):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        if n_env == 1:
            environment_seeds = [self.seed]
        else:
            environment_seeds = [random.randint(0, 1000000) for _ in range(n_env)]
        
        fitnesses = []
        
        for seed in environment_seeds:
            obs, _ = env.reset(seed=seed, initial_state=self.env_initial_state)
            fitness = 0.0
            
            while True:
                nn_inputs = env.process_observation(obs)
                nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])
                actions = (2 * nn_outputs - 1) * env.max_wheel_velocity

                obs, reward, done, truncated, _ = env.step(actions)
                fitness += reward

                if done or truncated:
                    break
            
            fitnesses.append(fitness)
        
        return np.mean(fitnesses)

    
    def _evaluate_genomes_batch(self, genomes, config, env):
        results = []
        
        for genome_id, genome in genomes:
            fitness = self._run_episode(genome, config, env, n_env = self.n_env)

            results.append((genome_id, fitness))
        
        return results

    def _calculate_fitness_neat(self, genomes, config):
        self.env.target_color = self.target_color

        if self.n_workers > 1:
            # ----- PARALLEL EVALUATION -----
            env_parallel = copy.deepcopy(self.env)
            batch_size = max(1, len(genomes) // self.n_workers)
            genome_batches = [genomes[i:i + batch_size] for i in range(0, len(genomes), batch_size)]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(self._evaluate_genomes_batch, batch, config, env_parallel) 
                           for batch in genome_batches]
                results = []
                for future in as_completed(futures):
                    results.extend(future.result())
            
            fitness_map = dict(results)
            for genome_id, genome in genomes:
                genome.fitness = fitness_map[genome_id]
            # -------------------------------
        else:
            # ----- SEQUENTIAL EVALUATION -----
            for genome_id, genome in genomes:
                genome.fitness = self._run_episode(genome, config, self.env, n_env = self.n_env)
            # -------------------------------
        
        # ----- REGULARIZATION -----
        if self.regularization_retention is not None and self.best_individual is not None:
            # Save max fitness without penalty
            self.fitness_no_penalty.append(max([genome.fitness for _, genome in genomes]))
            
            for _, genome in genomes:
                
                # Genetic distance penalty
                if self.regularization_retention == "gd" or self.regularization_retention == "genetic_distance":
                    penalty_distance = 0.0
                    config.compatibility_weight_coefficient = 0.6
                    config.compatibility_disjoint_coefficient = 1.0
                    penalty_distance += self.best_individual.distance(genome, config)

                    reg_penalty = self.reg_lambdas.get('gd') * penalty_distance
                
                # Weight protection penalty
                if self.regularization_retention == "wp" or self.regularization_retention == "weight_protection":
                    penalty_wp1 = 0.0
                    penalty_wp2 = 0.0
                    for c in genome.connections:
                        if c in self.best_individual.connections:
                            penalty_wp1 += (self.best_individual.connections[c].weight - genome.connections[c].weight) **2
                        else:
                            penalty_wp2 += genome.connections[c].weight ** 2

                    reg_penalty = self.reg_lambdas.get('wp')[0] * penalty_wp1 + self.reg_lambdas.get('wp')[1] * penalty_wp2

                # TODO: functional regularization

                genome.fitness -= reg_penalty # Apply the penalty
        # ---------------------------
        
        # ----- EVALUATE RETAINING -----
        if self.eval_retention is not None and self.prev_target_color is not None:
            # Evaluate genomes on the previous task
            if (self._current_generation % FREQUENCY_EVAL_RETENTION == 0 or 
                self._current_generation == self._generations - 1): # Evaluate at frequency and at the end of the evolution
                
                self.env.target_color = self.prev_target_color
                eval_genomes = copy.deepcopy(genomes)
                
                if "top" in self.eval_retention:
                    # Take top current genome
                    eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True)
                    top_genome = eval_genomes[0]
                    retention_top = self._run_episode(top_genome, config, self.env) 
                    print(f"Retention_top: {retention_top}")
                    self.retention_top_fitness.append(retention_top)
                    
                if "population" in self.eval_retention or "pop" in self.eval_retention: 
                    # Find the best genome on the previous task
                    for _, genome in eval_genomes:
                        genome.fitness = self._run_episode(genome, config, self.env)
                    eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True)
                    retention_pop = eval_genomes[0][1].fitness
                    print(f"Retention_pop: {retention_pop}")
                    self.retention_pop_fitness.append(retention_pop)
        # -------------------------------
        self._current_generation += 1
    
    def _calculate_fitness_deap(self, individual, n_env = 10):

        if n_env == 1:
            environment_seeds = [self.seed]
        else:
            environment_seeds = [random.randint(0, 1000000) for _ in range(n_env)]
        fitnesses = []
        
        for seed in environment_seeds:
            fitness = 0.0
            obs, _ = self.env.reset(seed=seed, initial_state=self.env_initial_state)
            
            self.controller_deap.set_weights_from_vector(individual) # Set the weights of the network
            
            while True:
                nn_inputs = self.env.process_observation(obs)
                nn_outputs = np.array(self.controller_deap.predict(nn_inputs))
                actions = (2 * nn_outputs - 1) * self.env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity
                    
                obs, reward, done, truncated, _ = self.env.step(actions)

                fitness += reward

                if done or truncated:
                    break
            
            fitnesses.append(fitness)

        return [float(np.mean(fitnesses))]
    
    def _run_neat(self, generations):
        self._generations = generations
        if self.config_neat is None:
            raise ValueError("Neat config object is not set. Set the path to the config file first.")

        stats = neat.StatisticsReporter()
        if self.population is None:
            self.population = neat.Population(self.config_neat)
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
        
        # Set up the fitness and individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox_deap = base.Toolbox()
        toolbox_deap.register("evaluate", self._calculate_fitness_deap)  # Evaluation function
        toolbox_deap.register("attr_float", random.uniform, -1.0, 1.0)  # Attribute generator
        toolbox_deap.register("individual", tools.initRepeat, creator.Individual,
                        toolbox_deap.attr_float, n=self.controller_deap.total_weights)  # Individual generator
        toolbox_deap.register("population", tools.initRepeat, list, toolbox_deap.individual)
        toolbox_deap.register("select", tools.selTournament, tournsize=max(1, int(self.population_size*0.03)))  # Selection function
        # toolbox.register("select", selElitistAndTournament)  # Selection function
        toolbox_deap.register("mate", tools.cxTwoPoint)  # Crossover function
        # toolbox.register("mate", tools.cxOnePoint)  # Crossover function
        # toolbox.register("mate", tools.cxUniform, indpb=0.5)  # 50% chance for each weight
        toolbox_deap.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # Mutation function

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("best", np.max)
        stats.register("avg", np.mean)
        stats.register("median", np.median)
        stats.register("std", np.std)
        stats.register("worst", np.min)
        
        self.population = toolbox_deap.population(n=self.population_size)  # Create a new population 
        
        hof = tools.HallOfFame(2)  # Hall of fame to store the best individual

        start = time.time()
        # Run the genetic algorithm
        self.population, log = algorithms.eaSimple(self.population, toolbox_deap, cxpb=0.8, mutpb=0.05,
                                                    ngen=generations, stats=stats, halloffame=hof, verbose=True)
        end = time.time()
        
        self.best_individual = hof[0]
        self.time_elapsed = end - start
        self.log = log

    def _run_cmaes(self, generations):
        if self.controller_deap is None:
            raise ValueError("DEAP controller is not set. Set the controller first.")
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox_deap = base.Toolbox()
        toolbox_deap.register("evaluate", self._calculate_fitness_deap)
        # Strategy parameters for CMA-ES
        strategy = cma.Strategy(centroid=[0.0]*self.controller_deap.total_weights, sigma=5.0, lambda_=self.population_size)
        toolbox_deap.register("generate", strategy.generate, creator.Individual)
        toolbox_deap.register("update", strategy.update)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("best", np.max)
        stats.register("avg", np.mean)
        stats.register("median", np.median)
        stats.register("std", np.std)
        stats.register("worst", np.min)

        hof = tools.HallOfFame(2)
        
        start = time.time()
        self.population, log = algorithms.eaGenerateUpdate(toolbox_deap, ngen=generations, stats=stats, halloffame=hof)
        end = time.time()
        
        self.best_individual = hof[0]
        self.time_elapsed = end - start
        self.log = log

    def _run_evostick(self, generations):
        if self.controller_deap is None:
            raise ValueError("DEAP controller is not set. Set the controller first.")
        
        n_elite = max(1, int(0.2 * self.population_size)) # 20% of the population will be copied to the next generation (elitism)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox_deap = base.Toolbox()
        toolbox_deap.register("evaluate", self._calculate_fitness_deap)  # Evaluation function
        toolbox_deap.register("attr_float", random.uniform, -1.0, 1.0)  # Attribute generator
        toolbox_deap.register("individual", tools.initRepeat, creator.Individual,
                        toolbox_deap.attr_float, n=self.controller_deap.total_weights)  # Individual generator
        toolbox_deap.register("population", tools.initRepeat, list, toolbox_deap.individual)
        toolbox_deap.register("select", tools.selBest, k=n_elite)  # Select top 20 individuals
        toolbox_deap.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=1.0)
        toolbox_deap.register("map", map)

        stats = tools.Statistics(lambda ind: ind.fitness.values) # Statistics to keep track of the evolution
        stats.register("best", np.max)
        stats.register("avg", np.mean)
        stats.register("median", np.median)
        stats.register("std", np.std)
        stats.register("worst", np.min)

        if self.population is None:
            self.population = toolbox_deap.population(n=self.population_size)  # Create a new population 
        
        hof = tools.HallOfFame(n_elite)  # Hall of fame to store the best individual

        start = time.time()
        # Run the genetic algorithm
        self.population, log = eaEvoStick(self.population, toolbox_deap, generations, stats=stats, halloffame=hof, verbose=True)
        end = time.time()

        self.best_individual = hof[0]
        self.time_elapsed = end - start
        self.log = log

    def run_best(self, on_prev_env = None, save = True, verbose = False):
        # TODO: make it prettier
        # TODO: check all this
        if self.best_individual is None:
            raise ValueError("Best individual is not set. Set it or tun the evolutionary algorithm first.")
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        if on_prev_env is not None and self.prev_target_color is None:
            raise ValueError("Previous target color is not set. Set it first.")
        if on_prev_env is not None and self.population is None:
            raise ValueError("Population is not set. Set it first.")
        if on_prev_env is not None and on_prev_env not in ["top", "population"]:
            raise ValueError("On previous environment must be one of: top, population")
        if on_prev_env is not None and self.evolutionary_algorithm != "neat":
            raise ValueError(f"Evaluation of retention available only for NEAT. Got {self.evolutionary_algorithm}")
        
        # Set up the neural network controller
        if self.evolutionary_algorithm == 'neat':
            if self.config_neat is None:
                raise ValueError("Neat config path is not set. Set the path to the config file first.")
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            if self.controller_deap is None:
                raise ValueError("DEAP controller is not set. Set the controller first.")
            # self.controller_deap.set_weights_from_vector(self.best_individual)
        
        genome_run = self.best_individual
        self.env.target_color = self.target_color

        if on_prev_env is not None:
            self.env.target_color = self.prev_target_color # Previous task
            
            if on_prev_env == "population":
                genomes = copy.deepcopy(list(self.population.population.items())) 
                
                # Find the best genome on the previous task
                for _, genome in genomes:
                    genome.fitness = self._run_episode(genome, self.config_neat, self.env)
                
                genomes.sort(key=lambda x: x[1].fitness, reverse=True)
                genome_run = genomes[0][1]
        
        if self.evolutionary_algorithm == 'neat':
            controller_neat = neat.nn.FeedForwardNetwork.create(genome_run, self.config_neat)
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            self.controller_deap.set_weights_from_vector(genome_run)

        frames = []
        done = False
        total_reward = 0
        obs, _ = self.env.reset(seed=self.seed, initial_state=self.env_initial_state)
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
        
        print(f"Total reward: {total_reward}")
        if save:
            if on_prev_env is not None:
                imageio.mimsave(f"results/{self.experiment_name}/episode_retention_{on_prev_env}.gif", frames, fps = 60)
            else:
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
        
        # TODO: check this and maybe change
        plot_evolution(bests, avgs = avgs, medians = medians, 
                       filename = f"results/{self.experiment_name}/evolution_plot.png")
        
        
        total_reward, info = self.run_best(save = True)
        
        if self.eval_retention is not None and self.prev_target_color is not None:
            total_reward_retention_top, info_retention_top = self.run_best(on_prev_env = "top", save = True)
            total_reward_retention_population, info_retention_population = self.run_best(on_prev_env = "population", save = True)
        else:
            total_reward_retention_top = None
            info_retention_top = None
            total_reward_retention_population = None
            info_retention_population = None

        # Stats
        logbook = {
            "best": bests,
            "avg": avgs,
            "median": medians,
            "std": stds,
            "retention_pop": self.retention_pop_fitness,
            "no_penalty": self.fitness_no_penalty,
        }
        # Experiment info
        experiment = {
            "name": self.experiment_name,
            "algorithm": self.evolutionary_algorithm,
            "generations":len(bests),
            "ep_duration": self.env.duration,
            "population_size": self.population_size,
            "target_color": self.target_color,
            "prev_target_color": self.prev_target_color,
            "agents": self.env.n_agents,
            "blocks": self.env.n_blocks,
            "distribution": self.env.distribution,
            "duration": self.env.duration,
            "n_colors": self.env.n_colors,
            "repositioning": self.env.repositioning,
            "blocks_in_line": self.env.blocks_in_line,
            "seed": self.seed,
            "best": bests[-1],
            "no_penalty": self.fitness_no_penalty[-1] if self.fitness_no_penalty else None,
            "retention_pop": self.retention_pop_fitness[-1] if self.retention_pop_fitness else None,
            "retention_top": self.retention_top_fitness[-1] if self.retention_top_fitness else None,
            "test_fitness": total_reward,
            "info": info,
            "correct_retrieves": len(info["correct_retrieves"]),
            "wrong_retrieves": len(info["wrong_retrieves"]),
            "test_fitness_retention_top": total_reward_retention_top,
            "info_retention_top": info_retention_top,
            "correct_retrieves_retention_top": len(info_retention_top["correct_retrieves"]) if info_retention_top is not None else None,
            "wrong_retrieves_retention_top": len(info_retention_top["wrong_retrieves"]) if info_retention_top is not None else None,
            "test_fitness_retention_population": total_reward_retention_population,
            "info_retention_population": info_retention_population,
            "correct_retrieves_retention_population": len(info_retention_population["correct_retrieves"]) if info_retention_population is not None else None,
            "wrong_retrieves_retention_population": len(info_retention_population["wrong_retrieves"]) if info_retention_population is not None else None,
            "retention_type": self.eval_retention,
            "regularization": self.regularization_retention,
            "regularization_lambdas": self.reg_lambdas,
            "time": self.time_elapsed,
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
            with open(f"results/{self.experiment_name}/neat_config.pkl", "wb") as f:
                pickle.dump(self.config_neat, f)
        elif self.evolutionary_algorithm in DEAP_ALGORITHMS:
            # Save the deap controller
            with open(f"results/{self.experiment_name}/controller.pkl", "wb") as f:
                pickle.dump(self.controller_deap, f)
    
    # TODO: change name of parameters
    def run(self, generations, n_workers = 1, eval_retention = None, regularization_retention = None):
        if self.name is None:
            raise ValueError("Name is not set. Set the name of the experiment first.")
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        if self.evolutionary_algorithm is None:
            raise ValueError("Evolutionary algorithm is not set. Set the evolutionary algorithm first.")
        if self.population_size is None:
            raise ValueError("Population size is not set. Set the population size first.")
        if eval_retention is not None:
            for e in eval_retention:
                if e not in ["top", "population"]:
                    raise ValueError("Evaluation of retention must be one of: top, population.")
        if eval_retention is not None and self.experiment_name is None:
            raise ValueError("Evaluation of retention not available for static environemnt (before drifts).")
        if regularization_retention is not None:
            if regularization_retention not in ["gd", "wp", "genetic_distance", "weight_protection"]:
                    raise ValueError("Regularization of retention must be one of: gd, wp, genetic_distance, weight_protection.")
        if regularization_retention is not None and self.experiment_name is None:
            raise ValueError("Regularization for retention not available for static environemnt (before drifts).")
        if (eval_retention is not None or regularization_retention is not None) and self.evolutionary_algorithm != "neat": 
            raise ValueError(f"Regularization/Evaluation of retention available only for NEAT. Got {self.evolutionary_algorithm}")
        
        available_cores = multiprocessing.cpu_count()
        self.n_workers = n_workers if n_workers <= available_cores else available_cores
        self.eval_retention = eval_retention
        self.regularization_retention = regularization_retention

        if self.experiment_name is None:
            distribution_str = "u" if self.env.distribution == "uniform" else "b"
            self.experiment_name = f"{self.name}" \
                f"/{self.evolutionary_algorithm}_{self.env.duration}_{generations}_{self.population_size}_{self.env.n_agents}_{self.env.n_blocks}_{distribution_str}" \
                f"/seed{self.seed}/static{self.target_color}" # First evolution (no drift - static)
        
        os.makedirs(f"results/{self.experiment_name}", exist_ok=True) # Create directory for the experiment results
    

        print(f"\n{self.experiment_name}")
        print(f"Running {self.evolutionary_algorithm} with with the following parameters:")
        print(f"Name: {self.name}")
        print(f"Duration of episode: {self.env.duration} in steps (one step is {environment.TIME_STEP} seconds)")
        print(f"Number of generations: {generations}")
        print(f"Population size: {self.population_size}")
        print(f"Number of agents: {self.env.n_agents}")
        print(f"Number of blocks: {self.env.n_blocks}")
        print(f"Distribution of blocks: {self.env.distribution}")
        print(f"Seed: {self.seed}")
        
        # Set the seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.env.reset(seed=self.seed, initial_state=self.env_initial_state)
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
