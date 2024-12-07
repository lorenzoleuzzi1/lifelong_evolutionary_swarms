results_dir="/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results"
reg_dir="reg_gd_general"
search_dir="$results_dir/$reg_dir"
echo "Searching in $search_dir"
# list all directiory in the directory
for dir in $(ls $search_dir); do
    echo $dir
    if [ -d "$search_dir/$dir" ]; then
        python3 plot_evolution.py "$reg_dir/$dir" -ret top
    fi
done