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
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 main.py neat easy 2 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 main.py ga easy 2 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 main.py cma-es easy 2 &
wait # Wait for all background jobs to finish
