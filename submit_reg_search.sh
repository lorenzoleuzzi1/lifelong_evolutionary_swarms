#!/bin/bash

#SBATCH --job-name=evoswarm
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1

worker_num=5

# Must be one less that the total number of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
echo ${nodes_array[@]}

search="coarse" # changable parameter
reg="gd" # changable parameter
distribution="u" # changable parameter

if [ "$search" == "coarse" ]; then

    if [ "$reg" == "gd" ]; then
        gd_lambda_range_u_coarse=(5 6 7 8 9 10 11)
        gd_lambda_range_b_coarse=(7 8 9 10 11 12 13)

        if [ "$distribution" == "u" ]; then
            param_space=("${gd_lambda_range_u_coarse[@]}")
        else
            param_space=("${gd_lambda_range_b_coarse[@]}")
        fi
    fi

    if [ "$reg" == "wp" ]; then
        wp_lambda_1_u_coarse=(0.001 0.01 0.1 1.0)
        wp_lambda_2_u_coarse=(0.0001 0.001 0.01 0.1)
        grid_wp_lambda_u_coarse=()

        for l1 in "${wp_lambda_1_u_coarse[@]}"; do
            for l2 in "${wp_lambda_2_u_coarse[@]}"; do
                if (( $(echo "$l1 > $l2" | bc -l) )); then
                    grid_wp_lambda_u_coarse+=("$l1 $l2")
                fi
            done
        done

        wp_lambda_1_b_coarse=(0.002 0.02 0.2 2.0)
        wp_lambda_2_b_coarse=(0.0002 0.002 0.02 0.2)
        grid_wp_lambda_b_coarse=()

        for l1 in "${wp_lambda_1_b_coarse[@]}"; do
            for l2 in "${wp_lambda_2_b_coarse[@]}"; do
                if (( $(echo "$l1 > $l2" | bc -l) )); then
                    grid_wp_lambda_b_coarse+=("$l1 $l2")
                fi
            done
        done

        if [ "$distribution" == "u" ]; then
            param_space=("${grid_wp_lambda_u_coarse[@]}")
        else
            param_space=("${grid_wp_lambda_b_coarse[@]}")
        fi
    fi
fi

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
    paramstr=$(echo $paramstr | tr '.' 'o') 
    
    srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} python3.11 run_and_drift.py --name $reg$search$paramstr --evo neat --steps 100 --generations 3 --population 100 --agents 5 --blocks 30 --seed 1 --distribution $distribution --regularization $reg --lambdas $param  --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[1]} python3.11 run_and_drift.py --name $reg$search$paramstr --evo neat --steps 100 --generations 3 --population 100 --agents 5 --blocks 30 --seed 1 --distribution $distribution --regularization $reg --lambdas $param  --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[2]} python3.11 run_and_drift.py --name $reg$search$paramstr --evo neat --steps 100 --generations 3 --population 100 --agents 5 --blocks 30 --seed 1 --distribution $distribution --regularization $reg --lambdas $param  --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[3]} python3.11 run_and_drift.py --name $reg$search$paramstr --evo neat --steps 100 --generations 3 --population 100 --agents 5 --blocks 30 --seed 1 --distribution $distribution --regularization $reg --lambdas $param  --workers 64 &
    srun --nodes=1 --ntasks=1 -w ${nodes_array[4]} python3.11 run_and_drift.py --name $reg$search$paramstr --evo neat --steps 100 --generations 3 --population 100 --agents 5 --blocks 30 --seed 1 --distribution $distribution --regularization $reg --lambdas $param  --workers 64 &
wait # Wait for all background jobs to finish
done


