#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1

worker_num=5

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name newgdreg --evo neat --steps 800 --generations 100 --population_size 300 --agents 5 --blocks 30 --seed 1 --distribution uniform &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name newgdreg --evo neat --steps 800 --generations 100 --population_size 300 --agents 5 --blocks 30 --seed 2 --distribution uniform &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name newgdreg --evo neat --steps 800 --generations 100 --population_size 300 --agents 5 --blocks 30 --seed 3 --distribution uniform &
srun --nodes=1 --ntasks=1 -w ${nodes_array[3]} python3.11 run_and_drift.py --name newgdreg --evo neat --steps 800 --generations 100 --population_size 300 --agents 5 --blocks 30 --seed 4 --distribution uniform &
srun --nodes=1 --ntasks=1 -w ${nodes_array[4]} python3.11 run_and_drift.py --name newgdreg --evo neat --steps 800 --generations 100 --population_size 300 --agents 5 --blocks 30 --seed 5 --distribution uniform &
wait # Wait for all background jobs to finish