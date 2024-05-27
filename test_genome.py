import neat
import utils
import numpy as np
from neural_controller import NeuralController
from environment import SwarmForagingEnv
import argparse
import pickle
import imageio

DEAP_ALGORITHMS = ['cma-es', 'ga', 'evostick']
def run_episode(env : SwarmForagingEnv, genome, algorithm, n_steps=500, verbose = False):
    obs = env.reset()
    done = False
    total_reward = 0
    
    # Set up the neural network controller
    if algorithm == 'neat':
        config_path = "./config-feedforward.txt"
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        
        config.genome_config.add_activation('neat_sigmoid', utils.neat_sigmoid)
        nn_controller = neat.nn.FeedForwardNetwork.create(genome, config)
    elif algorithm in DEAP_ALGORITHMS:
        input_dim = (env.n_types + 2 + 1) * env.n_neighbors + 2 + env.n_types - 2
        output_dim = 3
        hidden_units = [16]
        layer_sizes = [input_dim] + hidden_units + [output_dim]
        nn_controller = NeuralController(layer_sizes, hidden_activation="neat_sigmoid", output_activation="neat_sigmoid")
        nn_controller.set_weights_from_vector(genome)
    
    frames = []
    obs, _ = env.reset()
    frames.append(env.render(verbose))
    
    for i in range(n_steps):
        inputs = env.process_observation(obs)
        
        if algorithm == 'neat':
            outputs = np.array([nn_controller.activate(input) for input in inputs])
        elif algorithm in DEAP_ALGORITHMS:
            outputs = np.array(nn_controller.predict(inputs))

        actions = (2 * outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity

        obs, reward, done, _, info = env.step(actions)
        frames.append(env.render(verbose))
        total_reward += reward

        if done:
            break
    
    return total_reward, frames, info

def main(folder_path):
    # Load env 
    with open(folder_path + '/env.pkl', 'rb') as f:
        env = pickle.load(f)
    # Load the best genome
    with open(folder_path + '/best_genome.pkl', 'rb') as f:
        genome = pickle.load(f)
    
    if 'neat' in folder_path:
        algorithm = 'neat'
    elif 'cma-es' in folder_path or 'ga' in folder_path or 'evostick' in folder_path:
        algorithm = 'deap'
    else:
        raise ValueError('Invalid algorithm')
    
    total_reward, frames = run_episode(env, genome, algorithm)
    print(f"Total reward: {total_reward}")

    # Save the frames as a GIF
    imageio.mimsave(f'results/{folder_path}/best_episode.gif', frames, fps=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Test evolutionary swarm', 
        description='Test a controller produced from an evolutionary algorithm on the swarm foraging environment.')
    parser.add_argument('path', type=str, help='Path of the experiment directory')
    args = parser.parse_args()
    main(args.path)
