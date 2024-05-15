from run import run_neat, run_ga, run_cmaes
import argparse
import numpy as np
import environment

AVAILABLE_SCRIPTS = ["ga", "cma-es", "neat"]
AVAILABLE_INITIAL_SETTINGS = ["easy", "medium", "hard", "custom"]

def main(name, 
        script, 
        n_steps,
        n_generations,
        population_size, 
        initial_setting_type, 
        n_agents = None, 
        n_blocks = None, 
        seed = 0):
    
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

    if script == "neat":
        run_neat(name, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed)
    elif script == "ga":
        run_ga(name, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed)
    else:
        run_cmaes(name, n_generations, n_steps, population_size, initial_setting, n_agents, n_blocks, seed)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input for evolutionary swarm')
    parser.add_argument(
        'name', type=str,
        help=f'The name of the experiment.'
        )
    parser.add_argument(
        'script', type=str,
        help=f'The name of the experiment. \ Must be one of: {AVAILABLE_SCRIPTS}'
        )
    parser.add_argument(
        'n_steps', type=int,
        help='The number of steps of each episode.'
        )
    parser.add_argument(
        'n_generations', type=int,
        help='The number of generations to run the algorithm.'
        )
    parser.add_argument(
        'population_size', type=int,
        help='The size of the population for the evolutionary algorithm.'
        )
    parser.add_argument(
        'initial_setting', type=str,
        help=f'The initial setting of the environment. \ Must be one of: {AVAILABLE_INITIAL_SETTINGS}'
        )
    # Optional arguments
    parser.add_argument(
        '--n_agents', type=int, default=None,
        help='The number of agents in the arena.'
        )
    parser.add_argument(
        '--n_blocks', type=int, default=None,
        help='The number of blocks in the arena.'
        )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='The seed for the random number generator.'
        )
    args = parser.parse_args()
    
    # TODO: reorder the arguments
    
    if args.script not in AVAILABLE_SCRIPTS:
        raise ValueError(f"Script must be one of {AVAILABLE_SCRIPTS}")
    if args.initial_setting not in AVAILABLE_INITIAL_SETTINGS:
        raise ValueError(f"Difficulty must be one of {AVAILABLE_INITIAL_SETTINGS}")
    if args.initial_setting == "custom" and (args.n_agents is None or args.n_blocks is None):
        raise ValueError(f"When using custom setting, n_agents and n_blocks must be specified.")
    if args.initial_setting != "custom" and (args.n_agents is not None or args.n_blocks is not None):
        print(f"When not using custom setting, n_agents and n_blocks are ignored.")
    
    print(f"Running {args.script} with {args.initial_setting} initial setting with the following params:")
    print(f"- Episode: {args.n_steps} steps, total time {args.n_steps * 0.1}s, 0.1s per step")
    print(f"- Number of generations: {args.n_generations}")
    print(f"- Population size: {args.population_size}")
    print(f"- Seed: {args.seed}\n")
    
    main(args.name, 
        args.script, 
        args.n_steps, 
        args.n_generations, 
        args.population_size,
        args.initial_setting,
        args.n_agents, 
        args.n_blocks, 
        args.seed)