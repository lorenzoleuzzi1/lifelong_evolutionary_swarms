#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1

worker_num=3

seeds=(0 1 2 3 4)
# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

coefficients=(0 5 10 15)
param=0

# You can print the param_space to verify
echo "Parameter Space search:"
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name reg_gd/$param --evo neat --steps 500 --generations 200 --population 500 --agents 5 --blocks 20 --colors 2 --seed 13 --distribution biased --regularization gd --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 20 --workers 64 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name reg_gd/$param --evo neat --steps 500 --generations 200 --population 500 --agents 5 --blocks 20 --colors 2 --seed 42 --distribution biased --regularization gd --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 20 --workers 64 &
srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name reg_gd/$param --evo neat --steps 500 --generations 200 --population 100 --agents 5 --blocks 20 --colors 2 --seed 69 --distribution biased --regularization gd --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 20 --workers 64 &
wait # Wait for all background jobs to finish
done



