#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1

worker_num=3

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

name="lifelong_evo"
generations=200
population=300
colors="3 4"
targets="3 4 3"
n_env=10

workers=64

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name $name --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --config $config --workers 64 --seed 0 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name $name --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --config $config --workers 64 --seed 1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name $name --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --config $config --workers 64 --seed 2 &
wait # Wait for all background jobs to finish
