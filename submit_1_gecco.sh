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
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name ll --generations 200 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 13 &
wait # Wait for all background jobs to finish
