import numpy as np
from scipy.stats import rv_discrete
from utils import *
from Instances import *
import matplotlib.pyplot as plt
import json

instances = sample_test_instances(num_instances=1, seed=42)

# Generate 1 test instance
instances = sample_test_instances(num_instances=1, seed=42)  # Set seed for reproducibility

# Print the generated instance
print(f"Number of jobs: {instances[0]['num_jobs']}\n")

# Print details of the first instance
for i in range(0,10):
    job = instances[0]['jobs'][i]
    print(f"{i+1}-th Job Details:")
    for key, value in job.items():
        print(f"  {key}: {value}")
    
with open('instances.json', 'w') as f:
    json.dump(instances, f)

grid, F_Lmax = compute_max_lateness_cdf(instances[0])

# ---- Plotting ----
t_max = 1000
plt.figure(figsize=(10, 5))
plt.plot(grid[:t_max], F_Lmax[:t_max], label="F_Lmax(t)")
plt.xlabel("Time t")
plt.ylabel("CDF")
plt.title("CDF of Maximum Lateness (t ≤ 1000)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

instance = instances[0]
scheduled_indices = [0, 1, 2]
grid, F_Lmax_LB = compute_lower_bound_max_lateness(instance, scheduled_indices)

t_max = 1000
plt.figure(figsize=(10, 5))
plt.plot(grid[:t_max], F_Lmax_LB[:t_max], label="Lower Bound CDF of Max Lateness")
plt.xlabel("Time t")
plt.ylabel("CDF")
plt.title("Lower Bound on Maximum Lateness (Partial Schedule)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
