#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

worker_num=1

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

name="ll"
generations=200
population=300
colors="3 4"
targets="3 4 3"
n_env=10
eval_retention="pop top"
workers=64

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name $name --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --workers 64 --seed 25 &
wait # Wait for all background jobs to finish

# Plotting
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 plot_evolution.py $name
if "pop" in eval_retention:
    srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 plot_evolution.py $name -r pop
if "top" in eval_retention:
    srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 plot_evolution.py $name -r top
