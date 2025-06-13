import numpy as np
from scipy.stats import rv_discrete
from utils import *
from Instances import *
import matplotlib.pyplot as plt
import json
import time

instances=process_file("test_10_01.txt")

# Print the generated instance
print(f"Number of jobs: {instances[3]['num_jobs']}\n")

# Print details of the first instance
for i in range(0,2):
    job = instances[2]['jobs'][i]
    print(f"{i+1}-th Job Details:")
    for key, value in job.items():
        print(f"  {key}: {value}")


instance = instances[8]

grid, F_Lmax = compute_max_lateness_cdf(instance)

grid1, F_Lmax_LB = compute_lower_bound_max_lateness_cdf(instance, scheduled_jobs=[])

grid2, F_Lmax_UB = compute_upper_bound_max_lateness_cdf(instance, scheduled_jobs=[])
# ---- Plotting ----
t_max = 3000
plt.figure(figsize=(10, 5))
plt.plot(grid[:t_max], F_Lmax[:t_max], label="Lateness")
plt.plot(grid1[:t_max], F_Lmax_UB[:t_max], label="Upper bound")
plt.plot(grid2[:t_max], F_Lmax_LB[:t_max], label="Lower bound")
plt.xlabel("Time t")
plt.ylabel("CDF")
plt.title("CDF of upper and lower bounds of Maximum Lateness ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

best_schedule = build_schedule_with_bounds(instance, threshold=0.975)