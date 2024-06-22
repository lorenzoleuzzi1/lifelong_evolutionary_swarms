#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1

worker_num=4

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run.py --name uni --evo neat --steps 500 --generations 300 --population_size 300 --agents 8 --blocks 30 --seed 1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run.py --name uni --evo cma-es --steps 500 --generations 300 --population_size 300 --agents 8 --blocks 30 --seed 1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run.py --name uni --evo ga --steps 500 --generations 300 --population_size 300 --agents 8 --blocks 30 --seed 1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run.py --name uni --evo evostick --steps 500 --generations 300 --population_size 300 --agents 8 --blocks 30 --seed 1 &
wait # Wait for all background jobs to finish
