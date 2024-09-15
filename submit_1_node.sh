#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

worker_num=1

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run.py --name d --evo cma-es --steps 500 --generations 50 --population_size 500 --agents 8 --blocks 20 --seed 99 &
wait # Wait for all background jobs to finish
