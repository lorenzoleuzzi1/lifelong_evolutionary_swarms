from run import run_neat, run_ga, run_cmaes
import argparse
import numpy as np
import environment

AVAILABLE_SCRIPTS = ["ga", "cma-es", "neat"]
AVAILABLE_INITIAL_SETTINGS = ["easy", "medium", "hard"]

def main(script, difficulty, n_generations):
    initial_setting = None
    if script not in AVAILABLE_SCRIPTS:
        raise ValueError(f"Script must be one of {AVAILABLE_SCRIPTS}")
    if difficulty not in AVAILABLE_INITIAL_SETTINGS:
        raise ValueError(f"Difficulty must be one of {AVAILABLE_INITIAL_SETTINGS}")
    
    if difficulty == "easy":
        initial_setting = {
            'agents': np.array([[0, 5], [0, 10], [0, 15]], dtype=float),
            'blocks': np.array([[9, 16], [10, 12], [11, 6]], dtype=float),
            'colors': np.array([environment.RED, environment.GREEN, environment.RED], dtype=int)
            }
    elif difficulty == "medium":
        initial_setting = {
            'agents': np.array([[0, 5], [0, 10], [0, 15]], dtype=float),
            'blocks': np.array([[9, 16], [13, 7], [6, 5], [10, 11], [9, 7]], dtype=float),
            'colors': np.array([environment.RED, environment.RED, environment.BLUE, environment.GREEN, environment.RED], dtype=int)
            }
    else:
        raise ValueError("Not implemented")
    
    filename = script + "_" + difficulty + "_" + str(n_generations) + ".png"
    if script == "neat":
        run_neat(initial_setting, n_generations, filename)
    elif script == "ga":
        run_ga(initial_setting, n_generations, filename)
    else:
        run_cmaes(initial_setting, n_generations, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input for evolutionary swarm')
    parser.add_argument(
        'script', type=str,
        help=f'The name of the experiment. \ Must be one of: {AVAILABLE_SCRIPTS}'
        ) 
    parser.add_argument(
        'difficulty', type=str,
        help=f'The initial setting of the environment. \ Must be one of: {AVAILABLE_INITIAL_SETTINGS}'
        )
    parser.add_argument(
        'n_generations', type=int,
        help='The number of generations to run the algorithm'
        )
    args = parser.parse_args()
    
    if args.script is None:
        args.script = "neat"
    if args.difficulty is None:
        args.difficulty = "easy"
    if args.n_generations is None:
        args.n_generations = 100
    
    main(args.script, args.difficulty, args.n_generations)