import matplotlib.pyplot as plt
import numpy as np

def plot_data(bests, avgs = None, medians = None, stds = None, completion_fitness = None):
    x_values = np.arange(len(np.array(avgs)))
    plt.plot(bests, label="best")
    
    if stds is not None and avgs is not None:
        plt.errorbar(x_values, avgs, yerr=stds, label='avg +- std')
    if avgs is not None and stds is None:
        plt.plot(avgs, label='avg')
    if medians is not None:
        plt.plot(medians, label='median', color='purple')
    if completion_fitness is not None:
        plt.axhline(y=completion_fitness, color='g', linestyle='--', label='completion criterion')
    
    plt.legend()
    plt.title("Fitness over generations")
    plt.show()
