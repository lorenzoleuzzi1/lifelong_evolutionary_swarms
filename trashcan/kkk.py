import time
import copy
from environment import SwarmForagingEnv

env = SwarmForagingEnv(n_agents=5, n_blocks=30, target_color=3,
                        duration=800, distribution="uniform",
                        repositioning=True,
                        max_retrieves=10)

# Measure the time for redefining
start_time = time.time()
for i in range(1000):
    env1 = SwarmForagingEnv(n_agents=5, n_blocks=30, target_color=3,
                            duration=800, distribution="uniform",
                            repositioning=True,
                            max_retrieves=10)
print("Redefining time:", time.time() - start_time)

# Measure the time for deep copy
start_time = time.time()
for i in range(1000):
    env2 = copy.deepcopy(env)
print("Deep copy time:", time.time() - start_time)
