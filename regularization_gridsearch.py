from environment import SwarmForagingEnv, RED, BLUE
from experiment import EvoSwarmExperiment
from neural_controller import NeuralController

def main(name, 
        script, 
        steps,
        generations,
        population_size,
        n_agents, 
        n_blocks,
        distribution,
        seed,
        workers, 
        drifts,
        w, i, d):
    
    env = SwarmForagingEnv(n_agents = n_agents, n_blocks = n_blocks, target_color=drifts[0],
                           duration=steps, distribution=distribution)
    
    controller_deap = None
    config_path_neat = None
    if script == "neat":
        config_path_neat = "config-feedforward-fixed.txt"
    else:
        input_dim = (env.n_types + 2 + 1) * env.n_neighbors + 2 + env.n_types - 2
        output_dim = 3
        hidden_units = [4]
        layer_sizes = [input_dim] + hidden_units + [output_dim]
        controller_deap = NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")
    
    experiment = EvoSwarmExperiment(env = env, name = name, evolutionary_algorithm=script, population_size=population_size, 
                                    controller_deap=controller_deap, config_path_neat=config_path_neat, seed = seed,
                                    regularization_lambdas=[w, i, d])
    
    experiment.run(generations, n_workers = workers)
    
    for drift in drifts[1:]:
        experiment.change_objective(drift)
        experiment.run(generations, n_workers = workers,
                       eval_retaining = "top", regularization_retaining = ["weight", "innovation", "distance"])

if __name__ == "__main__":
    lambda_w = [0]
    lambda_i = [0]
    lambda_d = [0.01, 0.1, 1, 5]
    size_grid = len(lambda_w) * len(lambda_i) * len(lambda_d)
    print(f"Grid search with {size_grid} configurations")

    drifts = [RED, BLUE, RED]
    seed = 1
    # grid search
    for d in lambda_d:
        print(f"Running with lambda_d = {d}")
        main(f"dreg_{d}", "neat", 50, 2, 50, 5, 30, "uniform", 1, 32, drifts, 0, 0, d)
        main(f"dreg_{d}", "neat", 50, 2, 50, 5, 30, "biased", 1, 32, drifts, 0, 0, d)
