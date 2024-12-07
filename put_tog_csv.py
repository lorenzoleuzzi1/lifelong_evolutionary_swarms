import os
import pandas as pd

results_dir = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results"
reg_dir = "reg_gd_general"
distribution = "_b"

def put_to_csv(results_dir, reg_dir):
    csvs = []
    for exp_dir in os.listdir(f"{results_dir}/{reg_dir}"):
        exp_dir_path = f"{results_dir}/{reg_dir}/{exp_dir}"
        if os.path.isdir(exp_dir_path):  # Check if it's a directory
            for exp_dir_d in os.listdir(exp_dir_path):
                exp_dir_d_path = f"{exp_dir_path}/{exp_dir_d}"
                if exp_dir_d.endswith(distribution) and os.path.isdir(exp_dir_d_path):  # Check if it's a directory
                    for file in os.listdir(exp_dir_d_path):
                        if file.endswith(".csv"):
                            csvs.append((file, exp_dir, exp_dir_d))  # Fix this append to be a tuple
    
    # Put together the csvs in a single dataframe saving only the last line of each csv
    df = pd.DataFrame()  # Initialize an empty DataFrame
    csvs_list = []

    for csv in csvs:
        df_csv = pd.read_csv(f"{results_dir}/{reg_dir}/{csv[1]}/{csv[2]}/{csv[0]}")
        df_csv["Name"] = csv[1]
        
        # Use pd.concat() to append the last row
        csvs_list.append(df_csv.iloc[[-1]])  # iloc[[-1]] selects the last row as a DataFrame
    # Order by name
    csvs_list.sort(key=lambda x: x["Name"].values[0])
    # Concatenate all the dataframes at once after the loop
    df = pd.concat(csvs_list, ignore_index=True)

    df.to_csv(f"{results_dir}/{reg_dir}_together{distribution}.csv")

put_to_csv(results_dir, reg_dir)


    