name="snz2025/reg_gd_snz_0"
param1=0
param2=7.5
param3=11
param4=12.5
param5=20

python3 plot_evolution.py $name/$param1 
python3 plot_evolution.py $name/$param1 -r pop
python3 plot_evolution.py $name/$param1 -r top
python3 plot_evolution.py $name/$param2 -r top
# python3 plot_evolution.py $name/$param3 -r top
# python3 plot_evolution.py $name/$param4 -r top
# python3 plot_evolution.py $name/$param5 -r top