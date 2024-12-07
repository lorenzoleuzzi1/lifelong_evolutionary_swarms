import pandas as pd
import matplotlib.pyplot as plt

distribution = "b"
distribution_string = "Uniform" if distribution == "u" else "Biased"
reg = "gd_general"
results_dir = "/Users/lorenzoleuzzi/Library/CloudStorage/OneDrive-UniversityofPisa/lifelong_evolutionary_swarms/results"

# Read csv
df = pd.read_csv(f"{results_dir}/reg_{reg}_together_{distribution}.csv")

coefficient = df["Name"]
if reg == "wp":
    # Since we have two coefficient in the form coef1_coef2, we need to split them
    # and take the norm of the vector
    coefficient = coefficient.apply(lambda x: x.split("_"))
    coefficient = coefficient.apply(lambda x: (float(x[0])**2 + float(x[1])**2)**0.5)

evo2_reef = df["evo2_reef"]
evo3_reef = df["evo3_reef"]
avg_ref = df["avg_reef"]

# Calculate impact
# Correletion between coefficient and evo2_reef
correlation_2 = coefficient.corr(evo2_reef)
correlation_3 = coefficient.corr(evo3_reef)
correlation_avg = coefficient.corr(avg_ref)
print(f"Correlation between lambda {reg} and evo2_reef: {correlation_2}")
print(f"Correlation between lambda {reg} and evo3_reef: {correlation_3}")
print(f"Correlation between lambda {reg} and avg_ref: {correlation_avg}")

# Plot the impact
if reg == "gd" or reg == "gd_general":
    x_label = "$\lambda_{" +f"{reg}" "}$"
else:
    x_label = "$|| \lambda_{" +f"{reg}" "}||$"

plt.scatter(coefficient, evo2_reef)
plt.xlabel(x_label, fontsize=14)
plt.ylabel("REF$_2$", fontsize=14)
#plt.title(f"Correlation between " + x_label + f" and REF$_2$: {round(correlation_2, 2)}, {distribution_string}")
plt.savefig(f"{results_dir}/impact_{reg}_evo2_{distribution}.png", bbox_inches='tight')
plt.show()

plt.scatter(coefficient, evo3_reef)
plt.xlabel(x_label, fontsize=14)
plt.ylabel("REF$_3$", fontsize=14)
#plt.title(f"Correlation between " + x_label + f" and REF$_3$: {round(correlation_3, 2)}, {distribution_string}")
plt.savefig(f"{results_dir}/impact_{reg}_evo3_{distribution}.png", bbox_inches='tight')
plt.show()

plt.scatter(coefficient, avg_ref)
plt.xlabel(x_label, fontsize=14)
plt.ylabel("AvgREF", fontsize=14)
# plt.title(f"Correlation between " + x_label + f" and AvgREF: {round(correlation_avg, 2)}, {distribution_string}")
plt.savefig(f"{results_dir}/impact_{reg}_avg_{distribution}.png", bbox_inches='tight')
plt.show()