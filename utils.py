import matplotlib.pyplot as plt
import numpy as np
import neat
import environment
import neural_controller
from deap import tools

def plot_data(bests, avgs = None, medians = None, stds = None, completion_fitness = None, filename = None):
    x_values = np.arange(len(np.array(avgs)))
    plt.plot(bests, label="best")
    
    if stds is not None and avgs is not None:
        plt.errorbar(x_values, avgs, yerr=stds, label='avg +- std', alpha=0.6)
    if avgs is not None and stds is None:
        plt.plot(avgs, label='avg', alpha=0.6)
    if medians is not None:
        plt.plot(medians, label='median', color='purple', alpha=0.6)
    if completion_fitness is not None:
        plt.axhline(y=completion_fitness, color='g', linestyle='--', label='completion criterion')
    
    plt.ylim([-100, 100])
    plt.legend()
    plt.title("Fitness over generations")
    
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def selElitistAndTournament(individuals, k, frac_elitist = 0.1, tournsize = 3):
    # TODO: not really elitism
    return tools.selBest(individuals, int(k*frac_elitist)) + tools.selTournament(individuals, int(k*(1-frac_elitist)), tournsize=tournsize)

def inverse_sigmoid(y):
    return np.log(y / (1 - y))

def neat_sigmoid(x):
    return 1 / (1 + np.exp(-4.9 * x))

