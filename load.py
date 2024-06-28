from experiment import EvoSwarmExperiment
def main(path):
    e = EvoSwarmExperiment()
    e.load(path)
    e.run_best()
    
if __name__ == '__main__':
    path = "results/driftpun3_neat_800_300_300_8_20_40_drift34"
    main(path)
    path = "results/driftpun3u_neat_800_300_300_8_20_40_drift34"
    main(path)
    path = "results/driftpun3_neat_800_300_300_8_20_40_drift34_drift43"
    main(path)
    path = "results/driftpun3u_neat_800_300_300_8_20_40_drift34_drift43"
    main(path)