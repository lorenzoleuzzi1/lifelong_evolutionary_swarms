#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1

worker_num=5

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

name="red_green_red"
reg="gd" # wp, gd, or fun
param=11
generations=200
population=300
n_env=10
eval_retention="top"
config="config-feedforward.txt"
moredrifts=false
workers=64
seed1=13
seed2=17
seed3=24
seed4=31
seed5=42

# Launching different instances of the python script with different parameters
echo "Launching Python scripts with different parameters:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run.py --name $name --generations $generations --population $population --evals $n_env --eval_retention $eval_retention --regularization $reg --lambd $param --config $config --moredrifts $moredrifts --workers $workers --seed $seed1 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run.py --name $name --generations $generations --population $population --evals $n_env --eval_retention $eval_retention --regularization $reg --lambd $param --config $config --moredrifts $moredrifts --workers $workers --seed $seed2 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run.py --name $name --generations $generations --population $population --evals $n_env --eval_retention $eval_retention --regularization $reg --lambd $param --config $config --moredrifts $moredrifts --workers $workers --seed $seed3 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[3]} python3.11 run.py --name $name --generations $generations --population $population --evals $n_env --eval_retention $eval_retention --regularization $reg --lambd $param --config $config --moredrifts $moredrifts --workers $workers --seed $seed4 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[4]} python3.11 run.py --name $name --generations $generations --population $population --evals $n_env --eval_retention $eval_retention --regularization $reg --lambd $param --config $config --moredrifts $moredrifts --workers $workers --seed $seed5 &
wait # Wait for all background jobs to finish
