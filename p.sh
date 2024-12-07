#!/bin/bash

python3 plot_evolution.py evo_algos 
python3 plot_evolution.py baseline_blue 
python3 plot_evolution.py baseline_red
python3 plot_evolution.py drift 
python3 plot_evolution.py drift -ret top
python3 plot_evolution.py drift -ret pop
python3 plot_evolution.py evo_algos_general 
python3 plot_evolution.py baseline_blue_general
python3 plot_evolution.py baseline_red_general
python3 plot_evolution.py drift_general
python3 plot_evolution.py drift_general -ret top
python3 plot_evolution.py drift_general -ret pop


