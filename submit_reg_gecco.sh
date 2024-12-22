#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1

worker_num=5

seeds=(0 1 2 3 4)
# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

reg="gd" # changable parameter
distribution="u"


param_space=(0 5 10 15)

# print the param_space 
echo "Parameter Space search:"
echo ${#param_space[@]}
for param in "${param_space[@]}"; do
    echo "$param"
done

if [ "$distribution" == "u" ]; then
    distribution="uniform"
fi
if [ "$distribution" == "b" ]; then
    distribution="biased"
fi

# You can print the param_space to verify
echo "Parameter Space search:"
for param in "${param_space[@]}"; do
    echo "$param"
    paramstr=$(echo $param | tr ' ' '_') 
    # paramstr=$(echo $paramstr | tr '.' 'o') 

    srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name reg_g_$reg/$paramstr --evo neat --steps 500 --generations 200 --population 500 --agents 5 --blocks 20 --colors 4 --seed ${seeds[1]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 10 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name reg_g_$reg/$paramstr --evo neat --steps 500 --generations 200 --population 500 --agents 5 --blocks 20 --colors 4 --seed ${seeds[1]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 10 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name reg_g$reg/$paramstr --evo neat --steps 500 --generations 200 --population 100 --agents 5 --blocks 20 --colors 4 --seed ${seeds[2]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 10 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[3]} python3.11 run_and_drift.py --name reg_g$reg/$paramstr --evo neat --steps 500 --generations 100 --population 100 --agents 5 --blocks 20 --colors 4 --seed ${seeds[3]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 10 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[4]} python3.11 run_and_drift.py --name reg_g$reg/$paramstr --evo neat --steps 500 --generations 100 --population 100 --agents 5 --blocks 20 --colors 4 --seed ${seeds[4]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 10 --workers 64 &
wait # Wait for all background jobs to finish
done



