#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1

worker_num=3 

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 main.py test15mag neat 500 300 200 custom --n_agents=10 --n_blocks=20 --seed=11 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 main.py test15mag ga 500 300 200 custom --n_agents=10 --n_blocks=20 --seed=11 &&
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 main.py test15mag cma-es 500 300 200 custom --n_agents=10 --n_blocks=20 --seed=11 & &
wait # Wait for all background jobs to finish
