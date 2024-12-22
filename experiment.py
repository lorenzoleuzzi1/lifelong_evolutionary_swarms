import neat.config
import environment
import neural_controller 
from utils import plot_evolution
import random
import numpy as np
import neat
import pickle
import time
import json
import os
import imageio
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

FREQUENCY_EVAL_RETENTION = 10

# TODO: dont save all the info for deap, just neat handles drifts
# TODO: create two separate classes and use checkpoints
# TODO: base class, DEAP class, NEAT class
# TODO: only NEAT handles drifts
# TODO: make a separte class only for NEAT

class LifelongEvoSwarmExperiment:

    def __init__(self,
                name : str = None,
                population_size : int = None,
                env : environment.SwarmForagingEnv = None,
                env_initial_state : dict = None,
                config_neat : neat.config.Config = None,
                reg_lambdas : dict = {"gd": 6.0, "wp": [0.4, 0.3], "functional": 0.5}, 
                seed : int = None,
                n_env : int = 1, # Number of environments for evaluating fitness
                n_workers : int = 1
                ):
        
        self.name = name
        self.population_size = population_size
        self.env = env
        self.env_initial_state = env_initial_state
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

        available_cores = multiprocessing.cpu_count()
        self.n_workers = n_workers if n_workers <= available_cores else available_cores

        self.n_env = n_env

        self.eval_retention = None
        self.regularization_retention = None
        
        self.logbook_generations = []
        self.logbook_summary = {
            "best": [],
            "no_penalty": [],
            "id_best": [],
            "avg": [],
            "median": [],
            "std": [],
            "retention_top": [],
            "id_retention_top": [],
            "retention_pop": [],
            "id_retention_pop": [],
        }
    
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
        with open(folder_path + '/neat_config.pkl', 'rb') as f:
            self.config_neat = pickle.load(f)

    
    def drift(self, new_target, env_initial_state = None):
        
        self.logbook_generations = []
        self.logbook_summary = {
            "best": [],
            "no_penalty": [],
            "id_best": [],
            "avg": [],
            "median": [],
            "std": [],
            "retention_top": [],
            "id_retention_top": [],
            "retention_pop": [],
            "id_retention_pop": [],
        }
        self.experiment_name = f"{self.experiment_name}_drift{self.target_color}{new_target}"
        self.prev_target_color = self.target_color
        self.target_color = new_target
        self._current_generation = 0
        
        self.env.target_color = new_target

        if env_initial_state is not None:
            self.env_initial_state = env_initial_state
    
    def _evaluate_genome(self, genome, config, env, regularization = None):
        
        if regularization not in [None, "genetic_distance", "weight_protection", "gd", "wp", "functional"]:
            raise ValueError("Regularization must be one of: genetic_distance, weight_protection, gd, wp, functional.")
        if regularization is not None and self.best_individual is None:
            raise ValueError("Best individual is not set. Run the evolutionary algorithm first.")

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        if regularization== "functional":
            prev_net = neat.nn.FeedForwardNetwork.create(self.best_individual, config)
            penalty_functional = 0.0
        
        if self.n_env == 1:
            environment_seeds = [self.seed]
        else:
            environment_seeds = [random.randint(0, 1000000) for _ in range(self.n_env)]
        
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

                if regularization == "functional":
                    prev_nn_outputs = np.array([prev_net.activate(nn_input) for nn_input in nn_inputs])
                    penalty_functional += np.sum(np.abs(nn_outputs - prev_nn_outputs))

                if done or truncated:
                    break

            fitnesses.append(fitness)
        
        
        # ----- REGULARIZATION -----
        # Genetic distance penalty
        if regularization == "gd" or self.regularization_retention == "genetic_distance":
            penalty_distance = 0.0
            config.compatibility_weight_coefficient = 0.6
            config.compatibility_disjoint_coefficient = 1.0
            penalty_distance += self.best_individual.distance(genome, config)

            penalty = self.reg_lambdas.get('gd') * penalty_distance
                
        # Weight protection penalty
        if self.regularization_retention == "wp" or self.regularization_retention == "weight_protection":
            penalty_wp1 = 0.0
            penalty_wp2 = 0.0
            for c in genome.connections:
                if c in self.best_individual.connections:
                    penalty_wp1 += (self.best_individual.connections[c].weight - genome.connections[c].weight) **2
                else:
                    penalty_wp2 += genome.connections[c].weight ** 2

            penalty = self.reg_lambdas.get('wp')[0] * penalty_wp1 + self.reg_lambdas.get('wp')[1] * penalty_wp2
        
                # Functional penalty
        if regularization == "functional":
            penalty = penalty_functional / len(environment_seeds) / env.duration
        # ---------------------------
        
        # The fitness is the mean over the various environments
        genome_fitness = np.mean(fitnesses)
        
        # Apply the penalty
        if regularization is not None:
            genome_penalized_fitness = genome_fitness - penalty
        else:
            genome_penalized_fitness = None
        
        return genome_fitness, genome_penalized_fitness

    def _evaluate_genomes_batch(self, genomes, config, env, regularization = None):
        results = []
        
        for genome_id, genome in genomes:
            fitness, penalized_fitness = self._evaluate_genome(genome, config, env, regularization)

            results.append((genome_id, [fitness, penalized_fitness]))
        
        return results

    def _evaluate_fitness(self, genomes, config):
        
        population_stats = {} # key: id, value: fitness, adjusted fitness, retention fitness
        
        self.env.target_color = self.target_color

        if self.n_workers > 1:
            # ----- PARALLEL EVALUATION -----
            env_parallel = copy.deepcopy(self.env)
            batch_size = max(1, len(genomes) // self.n_workers)
            genome_batches = [genomes[i:i + batch_size] for i in range(0, len(genomes), batch_size)]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(self._evaluate_genomes_batch, batch, config, env_parallel, self.regularization_retention) 
                        for batch in genome_batches]
                results = []
                for future in as_completed(futures):
                    results.extend(future.result())
            
            fitness_map = dict(results)
            for genome_id, genome in genomes:
                fitness, penalized_fitness = fitness_map[genome_id]
                if penalized_fitness is not None:
                    genome.fitness = penalized_fitness # If regularization is applied use penalized fitness
                    population_stats[genome_id] = [fitness, penalized_fitness] # fitness, adjusted fitness
                else:
                    genome.fitness = fitness
                    population_stats[genome_id] = [fitness] # fitness, adjusted fitness
            # -------------------------------
        else:
            # ----- SEQUENTIAL EVALUATION -----
            for genome_id, genome in genomes:
                fitness, penalized_fitness = self._evaluate_genome(genome, config, self.env, self.regularization_retention)
                if penalized_fitness is not None:
                    genome.fitness = penalized_fitness # If regularization is applied use penalized fitness
                    population_stats[genome_id] = [fitness, penalized_fitness] # fitness, adjusted fitness
                else:
                    genome.fitness = fitness
                    population_stats[genome_id] = [fitness] # fitness
            # -------------------------------
        
        best_genome = max(genomes, key=lambda x: x[1].fitness)
        self.logbook_summary["best"].append(best_genome[1].fitness)
        self.logbook_summary["id_best"].append(best_genome[0])
        
        if self.regularization_retention is not None:
            self.logbook_summary["no_penalty"].append(population_stats[best_genome[0]][1])
        
        # ----- EVALUATE RETENTION -----
        if self.eval_retention is not None and self.prev_target_color is not None:
            # Evaluate genomes on the previous task
            if (self._current_generation % FREQUENCY_EVAL_RETENTION == 0
                or self._current_generation == self._generations - 1): # Evaluate at frequency
                
                self.env.target_color = self.prev_target_color
                
                if "population" in self.eval_retention or "pop" in self.eval_retention:
                    eval_genomes = copy.deepcopy(genomes)
                    # Find the best genome on the previous task, reevaluate the population
                    if self.n_workers > 1:
                        # Parallel evaluation
                        # TODO: maybe dont repeat this code
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
                            retention_fitness, _ = fitness_map[genome_id]
                            genome.fitness = retention_fitness
                            population_stats[genome_id].append(retention_fitness) # add retention to stats
                    else:
                        # Sequential evaluation
                        for _, genome in eval_genomes:
                            retention_fitness, _ = self._evaluate_genome(genome, config, self.env)
                            genome.fitness = retention_fitness
                            population_stats[genome_id].append(retention_fitness) # add retention to stats

                    eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True) # Take the best genome for retention
                    id_retenion_pop = eval_genomes[0][0]
                    retention_pop = eval_genomes[0][1].fitness
                    self.logbook_summary["retention_pop"].append(retention_pop)
                    self.logbook_summary["id_retention_pop"].append(id_retenion_pop)
                    print(f"Retention_pop: {retention_pop}")

                if "top" in self.eval_retention:
                    eval_genomes = copy.deepcopy(genomes)
                    # Take top current genome and evaluate on the previous task
                    eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True)
                    id_top_genome = eval_genomes[0][0]
                    top_genome = eval_genomes[0][1]
                    retention_top, _ = self._evaluate_genome(top_genome, config, self.env) 
                    self.logbook_summary["retention_top"].append(retention_top)
                    self.logbook_summary["id_retention_top"].append(id_top_genome)     
                    print(f"Retention_top: {retention_top}")      
        # -------------------------------
        
        self.logbook_generations.append(population_stats)

        self._current_generation += 1
    
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
        self.best_individual = self.population.run(self._evaluate_fitness, generations)
        end = time.time()
        self.time_elapsed = end - start
        self.log = stats

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
        if on_prev_env is not None and on_prev_env not in ["top", "population", "pop"]:
            raise ValueError("On previous environment must be one of: top, population/pop.")
        if self.config_neat is None:
            raise ValueError("Neat config path is not set. Set the path to the config file first.")
        
        genome_run = self.best_individual
        self.env.target_color = self.target_color

        if on_prev_env is not None:
            self.env.target_color = self.prev_target_color # Previous task   

            if on_prev_env == "population":
                genomes = copy.deepcopy(list(self.population.population.items())) 
                
                # Find the best genome on the previous task
                for _, genome in genomes:
                    genome.fitness = self._evaluate_genome(genome, self.config_neat, self.env)
                
                genomes.sort(key=lambda x: x[1].fitness, reverse=True)
                genome_run = genomes[0][1]
        
        controller = neat.nn.FeedForwardNetwork.create(genome_run, self.config_neat)

        frames = []
        done = False
        total_reward = 0
        obs, _ = self.env.reset(seed=self.seed, initial_state=self.env_initial_state)
        frames.append(self.env.render(verbose))
        while True:
            inputs = self.env.process_observation(obs)
            
            outputs = np.array([controller.activate(input) for input in inputs])

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
        bests = self.log.get_fitness_stat(np.max)
        avgs = self.log.get_fitness_mean()
        medians = self.log.get_fitness_median()
        stds = self.log.get_fitness_stdev()        
        
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
        self.logbook_summary["avg"] = avgs
        self.logbook_summary["median"] = medians
        self.logbook_summary["std"] = stds

        # Experiment info
        experiment_info = {
            "name": self.experiment_name,
            "algorithm": "neat",
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
            "max_wheel_velocity": self.env.max_wheel_velocity,
            "sensor_range": self.env.sensor_range,
            "arena_size": self.env.size,
            "n_workers": self.n_workers,
            "n_env": self.n_env,
            "seed": self.seed,
            "best": bests[-1],
            "id_best": self.logbook_summary["id_best"][-1],
            "no_penalty": self.logbook_summary["no_penalty"][-1] if self.logbook_summary["no_penalty"] else None,
            "retention_top": self.logbook_summary["retention_top"][-1] if self.logbook_summary["retention_top"] else None,
            "retention_pop": self.logbook_summary["retention_pop"][-1] if self.logbook_summary["retention_pop"] else None,
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
        # Save the logbooks as json
        with open(f"results/{self.experiment_name}/logbook_summary.json", "w") as f:
            json.dump(self.logbook_summary, f, indent=4)
        with open(f"results/{self.experiment_name}/logbook_generations.json", "w") as f:
            json.dump(self.logbook_generations, f, indent=4)
        with open(f"results/{self.experiment_name}/logbook_species.json", "w") as f:
            json.dump(self.log.generation_statistics, f, indent=4)
        # Save experiment info as json
        with open(f"results/{self.experiment_name}/info.json", "w") as f:
            json.dump(experiment_info, f, indent=4)
        # Save the environment as pickle
        with open(f"results/{self.experiment_name}/env.pkl", "wb") as f:
            pickle.dump(self.env, f)
        # Save winner as pickle
        with open(f"results/{self.experiment_name}/best_genome.pkl", "wb") as f:
            pickle.dump(self.best_individual, f)
        # Save the population as pickle
        with open(f"results/{self.experiment_name}/population.pkl", "wb") as f:
            pickle.dump(self.population, f)
        # Save the neat config file
        with open(f"results/{self.experiment_name}/neat_config.pkl", "wb") as f:
            pickle.dump(self.config_neat, f)
        
    
    # TODO: change name of parameters
    def run(self, generations, eval_retention = None, regularization_retention = None):
        if self.name is None:
            raise ValueError("Name is not set. Set the name of the experiment first.")
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        if self.population_size is None:
            raise ValueError("Population size is not set. Set the population size first.")
        if eval_retention is not None:
            for e in eval_retention:
                if e not in ["top", "population", "pop"]:
                    raise ValueError("Evaluation of retention must be one of: top, population/pop.")
        if eval_retention is not None and self.experiment_name is None:
            raise ValueError("Evaluation of retention not available for static environemnt (before drifts).")
        if regularization_retention is not None:
            if regularization_retention not in ["gd", "wp", "genetic_distance", "weight_protection", "functional"]:
                    raise ValueError("Regularization of retention must be one of: gd, wp, genetic_distance, weight_protection, functional.")
        if regularization_retention is not None and self.experiment_name is None:
            raise ValueError("Regularization for retention not available for static environemnt (before drifts).")
        if regularization_retention is not None and self.reg_lambdas is None:
            raise ValueError("Regularization lambdas are not set. Set the regularization lambdas first.")
        if regularization_retention is not None and self.best_individual is None:
            raise ValueError("Best individual is not set. Run the evolutionary algorithm first (e.g. a static evolution to save a reference model).")
        
        self.eval_retention = eval_retention
        self.regularization_retention = regularization_retention

        if self.experiment_name is None:
            distribution_str = "u" if self.env.distribution == "uniform" else "b"
            self.experiment_name = f"{self.name}" \
                f"/neat_{self.env.duration}_{generations}_{self.population_size}_{self.env.n_agents}_{self.env.n_blocks}_{distribution_str}" \
                f"/seed{self.seed}/static{self.target_color}" # First evolution (no drift - static)
        
        os.makedirs(f"results/{self.experiment_name}", exist_ok=True) # Create directory for the experiment results
    
        print(f"\n{self.experiment_name}")
        print(f"Running neat with with the following parameters:")
        print(f"Name: {self.name}")
        print(f"Duration of episode: {self.env.duration} in steps (one step is {environment.TIME_STEP} seconds)")
        print(f"Number of generations: {generations}")
        print(f"Population size: {self.population_size}")
        print(f"Number of agents: {self.env.n_agents}")
        print(f"Number of blocks: {self.env.n_blocks}")
        print(f"Distribution of blocks: {self.env.distribution}")
        print(f"N colors: {self.env.n_colors}")
        print(f"Repositioning: {self.env.repositioning}")
        print(f"Blocks in line: {self.env.blocks_in_line}")
        print(f"Number of environments for evaluation: {self.n_env}")
        print(f"Number of workers: {self.n_workers}")
        print(f"Seed: {self.seed}")
        
        # Set the seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.env.reset(seed=self.seed, initial_state=self.env_initial_state)
        self.env.render() # Show the environment

        self._run_neat(generations)
        
        self._save_results()
