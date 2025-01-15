#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1

worker_num=5

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

name="drift343"
reg="gd" # wp, gd, or fun
param=15
generations=200
population=300
colors="3 4"
targets="3 4 3"
n_env=10
eval_retention="top"

workers=64

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name reg_${reg}_$name/$param --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --regularization $reg --lambdas $param --workers 64 --seed 17 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name reg_${reg}_$name/$param --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --regularization $reg --lambdas $param --workers 64 --seed 24 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name reg_${reg}_$name/$param --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --regularization $reg --lambdas $param --workers 64 --seed 42 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[3]} python3.11 run_and_drift.py --name reg_${reg}_$name/$param --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --regularization $reg --lambdas $param --workers 64 --seed 54 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[4]} python3.11 run_and_drift.py --name reg_${reg}_$name/$param --generations $generations --population $population --colors $colors --targets $targets --evals $n_env --eval_retention $eval_retention --regularization $reg --lambdas $param --workers 64 --seed 97 &
wait # Wait for all background jobs to finish
