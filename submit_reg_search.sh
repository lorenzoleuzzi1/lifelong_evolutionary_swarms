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
distribution="u" # changable parameter

if [ "$reg" == "gd" ]; then
        param_space=(1 2 3 4 5 6 7 8 9 10 11 12 13)
fi

if [ "$reg" == "wp" ]; then
    wp_lambda_1_coarse=(0.001 0.005 0.01 0.05 0.1 0.5 1.0)
    wp_lambda_2_coarse=(0.001 0.005 0.01 0.05 0.1 0.5 1.0)
    grid_wp_lambda_coarse=()

    for l1 in "${wp_lambda_1_coarse[@]}"; do
        for l2 in "${wp_lambda_2_coarse[@]}"; do
            if (( $(echo "$l1 > $l2" | bc -l) )); then
                grid_wp_lambda_coarse+=("$l1 $l2")
            fi
        done
    done

    param_space=("${grid_wp_lambda_coarse[@]}")
fi

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

    srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name reg_$reg/$paramstr --evo neat --steps 500 --generations 100 --population 300 --agents 5 --blocks 20 --colors 4 --seed ${seeds[0]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 1 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name reg_$reg/$paramstr --evo neat --steps 500 --generations 100 --population 300 --agents 5 --blocks 20 --colors 4 --seed ${seeds[1]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 1 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name reg_$reg/$paramstr --evo neat --steps 500 --generations 100 --population 300 --agents 5 --blocks 20 --colors 4 --seed ${seeds[2]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 1 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[3]} python3.11 run_and_drift.py --name reg_$reg/$paramstr --evo neat --steps 500 --generations 100 --population 300 --agents 5 --blocks 20 --colors 4 --seed ${seeds[3]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 1 --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[4]} python3.11 run_and_drift.py --name reg_$reg/$paramstr --evo neat --steps 500 --generations 100 --population 300 --agents 5 --blocks 20 --colors 4 --seed ${seeds[4]} --distribution $distribution --regularization $reg --lambdas $param  --targets 3 4 3 --eval_retention top --n_env 1 --workers 64 &
wait # Wait for all background jobs to finish
done



