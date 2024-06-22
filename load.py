from experiment import EvoSwarmExperiment
def main(path):
    e = EvoSwarmExperiment()
    e.load(path)
    e.run_best()
    
if __name__ == '__main__':
    path = "results/uni_neat_500_300_300_8_30_10"
    main(path)