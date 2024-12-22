#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=10
#SBATCH --tasks-per-node=1

worker_num=10

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 0 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 2 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[3]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 3 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[4]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 4 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[5]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 5 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[6]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 6 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[7]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 7 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[8]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 8 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[9]} python3.11 run_and_drift.py --name ll --generations 300 --population 500 --colors 2 --distribution biased --targets 3 4 3 --n_env 20 --eval_retention pop top --workers 64 --seed 9 &
wait # Wait for all background jobs to finish
