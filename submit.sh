#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1

worker_num=2

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_evolution.py --name whsim --script neat --n_steps 500 --n_generations 300 --population_size 500 --n_agents 8 --n_blocks 20 --seed 1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_evolution.py --name whsim --script cma-es --n_steps 500 --n_generations 300 --population_size 500 --n_agents 8 --n_blocks 20 --seed 1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_evolution.py --name whsim --script ga --n_steps 500 --n_generations 300 --population_size 500 --n_agents 8 --n_blocks 20 --seed 1 &
wait # Wait for all background jobs to finish
