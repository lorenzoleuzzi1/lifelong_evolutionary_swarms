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
                config_neat : neat.config.Config = None, 
                seed : int = None,
                n_envs : int = 1, # Number of environments for evaluating fitness
                n_workers : int = 1
                ):
        
        self.name = name
        self.population_size = population_size
        self.env = env
        self.config_neat = config_neat
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
        self.prev_target_colors = []

        available_cores = multiprocessing.cpu_count()
        self.n_workers = n_workers if n_workers <= available_cores else available_cores

        self.n_envs = n_envs

        self.eval_retention = None
        self.reg_type = None
        
        self.logbook_generations = []
        self.logbook_summary = {
            "best": [],
            "best_std": [],
            "no_penalty": [],
            "id_best": [],
            "avg": [],
            "median": [],
            "std": []
            # "retention_top": [],
            # "id_retention_top": [],
            # "retention_pop": [],
            # "id_retention_pop": [],
        }

        self.prev_models = []

        # Set the seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def load(self, folder_path):
        # TODO: check all of this... we might need to load more, but for now we are not using this
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
        self.steps = experiment["duration"]
        self.population_size = experiment["population_size"]
        self.experiment_name = experiment["name"]
        self.name = experiment["name"].split("_")[0]
        self.time_elapsed = experiment["time"]
        self.seed = experiment["seed"]
        self.target_color = experiment["target_color"]
        self.prev_target_colors = experiment["prev_target_colors"]
        # self.prev_target = experiment["target_color"]
        # Other loads
        with open(folder_path + '/neat_config.pkl', 'rb') as f:
            self.config_neat = pickle.load(f)

    def drift(self, new_target):
        # TODO: validate if we are calling this dirft method properly
        self.logbook_generations = []
        
        self.experiment_name = f"{self.experiment_name}_drift{self.target_color}{new_target}"
        self.prev_target_colors.append(self.target_color)
        self.target_color = new_target
        self._current_generation = 0
        
        self.env.target_color = new_target

        self.logbook_summary = {
            "best": [],
            "best_std": [],
            "no_penalty": [],
            "id_best": [],
            "avg": [],
            "median": [],
            "std": []
        }
        for prev_target in self.prev_target_colors:
            self.logbook_summary[f"retention_top_{prev_target}"] = []
            self.logbook_summary[f"id_retention_top_{prev_target}"] = []
            self.logbook_summary[f"retention_top_{prev_target}_std"] = []
            self.logbook_summary[f"retention_pop_{prev_target}"] = []
            self.logbook_summary[f"id_retention_pop_{prev_target}"] = [] 
            self.logbook_summary[f"retention_pop_{prev_target}_std"] = []

        self.prev_models.append(self.best_individual)    
    
    def _evaluate_genome(self, genome, config, env, env_seeds, regularization = None):
        
        if regularization not in [None, "genetic_distance", "weight_protection", "gd", "wp", "functional", "fun"]:
            raise ValueError("Regularization must be one of: genetic_distance, weight_protection, gd, wp, functional.")
        if regularization is not None and self.best_individual is None:
            raise ValueError("Best individual is not set. Run the evolutionary algorithm first.")
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        if regularization== "functional" or regularization == "fun":
            prev_nets = []
            penalty_functional = 0.0
            
            for prev_model in self.prev_models:
                prev_nets.append(neat.nn.FeedForwardNetwork.create(prev_model, config))
        
        fitnesses = []
        
        for seed in env_seeds:
            obs, _ = env.reset(seed=seed)
            fitness = 0.0
            
            while True:
                nn_inputs = env.process_observation(obs)
                nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])

                actions = (2 * nn_outputs - 1) * env.max_wheel_velocity

                obs, reward, done, truncated, _ = env.step(actions)

                fitness += reward

                if regularization == "functional" or regularization == "fun":
                    for prev_net in prev_nets:
                        prev_nn_outputs = np.array([prev_net.activate(nn_input) for nn_input in nn_inputs])
                        penalty_functional += np.sum(np.abs(nn_outputs - prev_nn_outputs))

                if done or truncated:
                    break

            fitnesses.append(fitness)
        
        
        # ----- REGULARIZATION -----
        # Genetic distance penalty
        if regularization == "gd" or regularization == "genetic_distance":
            penalty_distance = 0.0
            for prev_model in self.prev_models:
                config.compatibility_weight_coefficient = 0.6
                config.compatibility_disjoint_coefficient = 1.0
                penalty_distance += prev_model.distance(genome, config)           

            penalty = self.reg_lambda * penalty_distance
                
        # Weight protection penalty
        if regularization == "wp" or regularization == "weight_protection":
            penalty_wp1 = 0.0
            penalty_wp2 = 0.0
            for prev_model in self.prev_models:
                for c in genome.connections:
                    if c in self.best_individual.connections:
                        penalty_wp1 += (prev_model.connections[c].weight - genome.connections[c].weight) **2
                    else:
                        penalty_wp2 += genome.connections[c].weight ** 2    

            penalty = self.reg_lambda[0] * penalty_wp1 + self.reg_lambda[1] * penalty_wp2
        
        # Functional penalty
        if regularization == "functional" or regularization == "fun":
            penalty = self.reg_lambda * (penalty_functional / len(env_seeds) / env.duration)
        # ---------------------------
        
        # The fitness is the mean over the various environments
        genome_fitness = np.mean(fitnesses)
        std = np.std(fitnesses)
        
        # Apply the penalty
        if regularization is not None:
            genome_penalized_fitness = genome_fitness - penalty 
        else:
            genome_penalized_fitness = None
        
        return genome_fitness, genome_penalized_fitness, std # TODO !!!!

    def _evaluate_genomes_batch(self, genomes, config, env, env_seeds, regularization = None):
        results = []
        
        for genome_id, genome in genomes:
            fitness, penalized_fitness, std = self._evaluate_genome(genome, config, env, 
                                                               env_seeds, regularization)

            results.append((genome_id, [fitness, penalized_fitness, std]))
        
        return results

    def _evaluate_fitness(self, genomes, config):
        
        # --- Modify the genome to include task labels ---
        # Get a sample input to identify the target labels (last n_colors)
        example_input = self.env.process_observation(self.env.reset(None)[0])[0]
        
        # Indetify the missing connections
        missing_connections = []
        for i in range(len(example_input) - self.env.n_colors, len(example_input)):
            missing_connections.append((-(i+1), 0))
            missing_connections.append((-(i+1), 1))
            missing_connections.append((-(i+1), 2))

        for _, genome in genomes:
            mean_weight_value = sum([abs(c.weight) for c in genome.connections.values()]) / len(genome.connections)

            for connection in missing_connections:
                # If the connection is missing, add it as the mean value
                if connection not in genome.connections:
                    genome.add_connection(config.genome_config, connection[0], connection[1], mean_weight_value, True)
                else:
                    # If too small, set it to the min
                    if abs(genome.connections[connection].weight) < 0.1:
                        genome.connections[connection].weight = min(0.1, genome.connections[connection].weight)
                    # Enable it if it is disabled
                    if not genome.connections[connection].enabled:
                        genome.connections[connection].enabled = True
        # ----------------------------------------------

        # for genome_id, genome in genomes:
        #     if (-31, 0) not in genome.connections or (-31, 1) not in genome.connections or (-31, 2) not in genome.connections:
        #         print(f"Missing connections in {genome_id}") 
        #     if (-30, 0) not in genome.connections or (-30, 1) not in genome.connections or (-30, 2) not in genome.connections:
        #         print(f"Missing connections in {genome_id}") 

        population_stats = {} # key: id, value: fitness, adjusted fitness, retention fitness
        
        self.env.target_color = self.target_color

        if self.n_envs == 1:
            env_seeds = [self.seed]
        else:
            env_seeds = [random.randint(0, 1000000) for _ in range(self.n_envs)]

        if self.n_workers > 1:
            # ----- PARALLEL EVALUATION -----
            env_parallel = copy.deepcopy(self.env)
            batch_size = max(1, len(genomes) // self.n_workers)
            genome_batches = [genomes[i:i + batch_size] for i in range(0, len(genomes), batch_size)]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(self._evaluate_genomes_batch, batch, config, 
                                           env_parallel, env_seeds, self.reg_type) 
                        for batch in genome_batches]
                results = []
                for future in as_completed(futures):
                    results.extend(future.result())
            
            fitness_map = dict(results)
            for genome_id, genome in genomes:
                fitness, penalized_fitness, std = fitness_map[genome_id]
                if penalized_fitness is not None:
                    genome.fitness = penalized_fitness # If regularization is applied use penalized fitness
                    population_stats[genome_id] = {"fitness": fitness, "penalized_fitness": penalized_fitness, "std": std} 
                else:
                    genome.fitness = fitness
                    population_stats[genome_id] = {"fitness":fitness, "std": std} 
            # -------------------------------
        else:
            # ----- SEQUENTIAL EVALUATION -----
            for genome_id, genome in genomes:
                fitness, penalized_fitness, std = self._evaluate_genome(genome, config, self.env, env_seeds,
                                                                   self.reg_type)
                if penalized_fitness is not None:
                    genome.fitness = penalized_fitness # If regularization is applied use penalized fitness
                    population_stats[genome_id] = {"fitness": fitness, "penalized_fitness": penalized_fitness, "std": std}
                else:
                    genome.fitness = fitness
                    population_stats[genome_id] = {"fitness": fitness, "std": std}
            # -------------------------------
        
        best_genome = max(genomes, key=lambda x: x[1].fitness)
        self.logbook_summary["best"].append(best_genome[1].fitness)
        self.logbook_summary["id_best"].append(best_genome[0])
        self.logbook_summary["best_std"].append(population_stats[best_genome[0]]["std"])
        
        if self.reg_type is not None:
            self.logbook_summary["no_penalty"].append(population_stats[best_genome[0]]["fitness"])
        
        # ----- EVALUATE RETENTION -----
        if self.eval_retention is not None and self.prev_target_colors is not []:
            # Evaluate genomes on the previous task
            if (self._current_generation % FREQUENCY_EVAL_RETENTION == 0
                or self._current_generation == self._generations - 1): # Evaluate at frequency
                
                for prev_target in self.prev_target_colors:
                    self.env.target_color = prev_target
                    
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
                                futures = [executor.submit(self._evaluate_genomes_batch, batch, config,
                                                        env_parallel, env_seeds) 
                                        for batch in genome_batches]
                                results = []
                                for future in as_completed(futures):
                                    results.extend(future.result())
                            
                            fitness_map = dict(results)
                            for genome_id, genome in eval_genomes:
                                retention_fitness, _, retention_std = fitness_map[genome_id]
                                genome.fitness = retention_fitness
                                population_stats[genome_id][f"retention_{prev_target}"] = retention_fitness # add retention to stats
                                population_stats[genome_id][f"retention_{prev_target}_std"] = retention_std # add retention to stats
                        else:
                            # Sequential evaluation
                            for _, genome in eval_genomes:
                                retention_fitness, _ = self._evaluate_genome(genome, config, self.env, env_seeds)
                                genome.fitness = retention_fitness
                                population_stats[genome_id][f"retention_{prev_target}"] = retention_fitness # add retention to stats
                                population_stats[genome_id][f"retention_{prev_target}_std"] = retention_std # add retention to stats

                        eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True) # Take the best genome for retention
                        id_retenion_pop = eval_genomes[0][0]
                        retention_pop = eval_genomes[0][1].fitness
                        self.logbook_summary[f"id_retention_pop_{prev_target}"].append(id_retenion_pop)
                        self.logbook_summary[f"retention_pop_{prev_target}"].append(retention_pop)
                        self.logbook_summary[f"retention_pop_{prev_target}_std"].append(population_stats[id_retenion_pop][f"retention_{prev_target}_std"])
                        print(f"Retention_pop: {retention_pop}")

                    if "top" in self.eval_retention:
                        eval_genomes = copy.deepcopy(genomes)
                        # Take top current genome and evaluate on the previous task
                        eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True)
                        id_top_genome = eval_genomes[0][0]
                        top_genome = eval_genomes[0][1]
                        retention_top, _, retention_top_std = self._evaluate_genome(top_genome, config, self.env, env_seeds) 
                        self.logbook_summary[f"id_retention_top_{prev_target}"].append(id_top_genome) 
                        self.logbook_summary[f"retention_top_{prev_target}"].append(retention_top)
                        self.logbook_summary[f"retention_top_{prev_target}_std"].append(retention_top_std)
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
        # fitness_function(list(self.population.items()), self.config) # final evaluation
        self.time_elapsed = end - start
        self.log = stats

    def run_genome(self, id_genome, target_color, save = True, verbose = False):
        # TODO: make it prettier
        # TODO: check all this
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        if self.config_neat is None:
            raise ValueError("Neat config object is not set. Set the path to the config file first.")
        
        # Get the id genome from population
        genome = self.population.population[id_genome]
        
        self.env.target_color = target_color
        
        controller = neat.nn.FeedForwardNetwork.create(genome, self.config_neat)

        frames = []
        done = False
        total_reward = 0
        obs, _ = self.env.reset(seed=None)
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
        
        print(f"Reward: {total_reward}")
        
        if save:
            imageio.mimsave(f"results/{self.experiment_name}/episode_{id_genome}_{target_color}.gif", frames, fps = 60)
        
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
        
        # Stats
        self.logbook_summary["avg"] = avgs
        self.logbook_summary["median"] = medians
        self.logbook_summary["std"] = stds

        # Experiment info
        experiment_info = {
            "name": self.experiment_name,
            "time": self.time_elapsed,
            "algorithm": "neat",
            "generations":len(bests),
            "ep_duration": self.env.duration,
            "population_size": self.population_size,
            "target_color": self.target_color,
            "prev_target_color": self.prev_target_colors,
            "agents": self.env.n_agents,
            "blocks": self.env.n_blocks,
            "duration": self.env.duration,
            "n_colors": self.env.n_colors,
            "repositioning": self.env.repositioning,
            "blocks_in_line": self.env.blocks_in_line,
            "max_wheel_velocity": self.env.max_wheel_velocity,
            "sensor_range": self.env.sensor_range,
            "arena_size": self.env.size,
            "n_workers": self.n_workers,
            "n_env": self.n_envs,
            "seed": self.seed,
            "best": bests[-1],
            "id_best": self.logbook_summary["id_best"][-1],
            "no_penalty": self.logbook_summary["no_penalty"][-1] if self.logbook_summary["no_penalty"] else None,
            "retention_type": self.eval_retention,
            "regularization": self.reg_type,
            "regularization_lambdas": self.reg_lambda,
            # "retention_top": self.logbook_summary["retention_top"][-1] if self.logbook_summary["retention_top"] else None,
            # "retention_pop": self.logbook_summary["retention_pop"][-1] if self.logbook_summary["retention_pop"] else None,
            # "test_fitness": total_reward,
            # "info": info,
            # "correct_retrieves": len(info["correct_retrieves"]),
            # "wrong_retrieves": len(info["wrong_retrieves"])
        }
        for prev_target in self.prev_target_colors:
            experiment_info[f"id_retention_top_{prev_target}"] = self.logbook_summary[f"id_retention_top_{prev_target}"][-1] if self.logbook_summary[f"id_retention_top_{prev_target}"] else None
            experiment_info[f"retention_top_{prev_target}"] = self.logbook_summary[f"retention_top_{prev_target}"][-1] if self.logbook_summary[f"retention_top_{prev_target}"] else None
            experiment_info[f"id_retention_pop_{prev_target}"] = self.logbook_summary[f"id_retention_pop_{prev_target}"][-1] if self.logbook_summary[f"id_retention_pop_{prev_target}"] else None
            experiment_info[f"retention_pop_{prev_target}"] = self.logbook_summary[f"retention_pop_{prev_target}"][-1] if self.logbook_summary[f"retention_pop_{prev_target}"] else None
        
        test_stats = {}
        total_reward, info = self.run_genome(experiment_info["id_best"], self.target_color, save = True)
        test_stats["id"] = experiment_info["id_best"]
        test_stats["fitness"] = total_reward
        test_stats["correct_retrieves"] = len(info["correct_retrieves"])
        test_stats["wrong_retrieves"] = len(info["wrong_retrieves"])
        test_stats["info"] = info

        if self.eval_retention is not None and "top" in self.eval_retention:
            for prev_target in self.prev_target_colors:
                # this should be the same as best_individaul.. TODO: check
                total_reward, info = self.run_genome(experiment_info[f"id_retention_top_{prev_target}"], prev_target, save = True)
                test_stats[f"id_retention_top_{prev_target}"] = experiment_info[f"id_retention_top_{prev_target}"]
                test_stats[f"retention_top_{prev_target}"] = total_reward
                test_stats[f"retention_top_correct_retrieves_{prev_target}"] = len(info["correct_retrieves"])
                test_stats[f"retention_top_wrong_retrieves_{prev_target}"] = len(info["wrong_retrieves"])
                test_stats[f"retention_top_info_{prev_target}"] = info
        
        if self.eval_retention is not None and ("population" in self.eval_retention or "pop" in self.eval_retention):
            for prev_target in self.prev_target_colors:
                if experiment_info[f"id_retention_pop_{prev_target}"] not in self.population.population:
                    print(f"Genome {experiment_info[f'id_retention_pop_{prev_target}']} not in population.")
                    continue
                total_reward, info = self.run_genome(experiment_info[f"id_retention_pop_{prev_target}"], prev_target, save = True)
                test_stats[f"id_retention_pop_{prev_target}"] = experiment_info[f"id_retention_pop_{prev_target}"]
                test_stats[f"retention_pop_{prev_target}"] = total_reward
                test_stats[f"retention_pop_correct_retrieves_{prev_target}"] = len(info["correct_retrieves"])
                test_stats[f"retention_pop_wrong_retrieves_{prev_target}"] = len(info["wrong_retrieves"])
                test_stats[f"retention_pop_info_{prev_target}"] = info
        
        # Save the logbooks as json
        with open(f"results/{self.experiment_name}/logbook_summary.json", "w") as f:
            json.dump(self.logbook_summary, f, indent=4)
        with open(f"results/{self.experiment_name}/logbook_generations.json", "w") as f:
            json.dump(self.logbook_generations, f, indent=4)
        with open(f"results/{self.experiment_name}/logbook_species.json", "w") as f:
            json.dump(self.log.generation_statistics, f, indent=4)
        with open(f"results/{self.experiment_name}/test.json", "w") as f:
            json.dump(test_stats, f, indent=4)
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
        # Save prev models as pickle
        with open(f"results/{self.experiment_name}/prev_models.pkl", "wb") as f:
            pickle.dump(self.prev_models, f)
        # Save the neat config file
        with open(f"results/{self.experiment_name}/neat_config.pkl", "wb") as f:
            pickle.dump(self.config_neat, f)  
    
    # TODO: change name of parameters
    def run(self, generations : int, eval_retention : str = None, regularization_type : str = None, regularization_coefficient = None):

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
        if eval_retention is not None:
            if self.experiment_name is None or self.prev_target_colors is None:
                raise ValueError("Evaluation of retention not available for static environemnt (before drifts).")
        if regularization_type is not None:
            if regularization_coefficient is None:
                raise ValueError("Regularization coefficient is not set.")
            if regularization_type not in ["gd", "wp", "genetic_distance", "weight_protection", "functional", "fun"]:
                raise ValueError("Regularization of retention must be one of: gd, wp, genetic_distance, weight_protection, functional, fun.")
            if regularization_coefficient in ["gd", "genetic_distance"] and len(regularization_coefficient) != 1:
                raise ValueError("Genetic distance regularization must have one value.")
            if regularization_coefficient in ["wp", "weight_protection"] and len(regularization_coefficient) != 2:
                raise ValueError("Weight protection regularization must have two values.")
            if regularization_coefficient in ["functional", "fun"] and len(regularization_coefficient) != 1:
                raise ValueError("Functional regularization must have one value.")
            if self.experiment_name is None:
                raise ValueError("Regularization for retention not available for static environemnt (before drifts).")
            if self.best_individual is None:
                raise ValueError("Best individual is not set. Run the evolutionary algorithm first (i.e. a static evolution to save a reference model).")
        
        self.eval_retention = eval_retention
        self.reg_type = regularization_type
        self.reg_lambda = regularization_coefficient

        if self.experiment_name is None:
            self.experiment_name = f"{self.name}" \
                f"/neat_{self.env.duration}_{generations}_{self.population_size}_{self.env.n_agents}_{self.env.n_blocks}_{self.n_envs}" \
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
        print(f"Colors: {self.env.colors} (target color: {self.target_color}, n_colors: {self.env.n_colors})")
        # print(f"Repositioning: {self.env.repositioning}")
        # print(f"Blocks in line: {self.env.blocks_in_line}")
        print(f"Number of environments for evaluation: {self.n_envs}")
        print(f"Number of workers: {self.n_workers}")
        print(f"Seed: {self.seed}")

        self.env.reset(seed=self.seed)
        self.env.render() # Show the environment

        self._run_neat(generations)
        
        self._save_results()
