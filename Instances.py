import numpy as np
from scipy.stats import rv_discrete
from utils import *
import json

import numpy as np

def sample_test_instances(num_instances=1, seed=None):
    
    """Generates test instances including processing times, release times and due dates."""
    if seed:
        np.random.seed(seed)  # Set seed for reproducibility

    instances = []  # Store generated test instances

    for _ in range(num_instances):
        num_jobs = np.random.choice([10, 20])  # Randomly pick number of jobs
        
        job_data = []  # Store data for each job
        for _ in range(num_jobs):
            # Sample mu from the specified distributions
            if np.random.rand() < 0.5:
                mu = np.random.randint(50, 151)
            else:
                mu = np.random.randint(25, 76)

            # Sample CV from the specified distributions
            if np.random.rand() < 0.5:
                cv = np.random.uniform(1.4, 1.6)
            else:
                cv = np.random.uniform(0.4, 0.6)

            # Randomly assign skewness from the given values
            skew = np.random.choice([-0.5, 0, 0.5])

            # Solve for a1, a2, m (mode) using the existing method
            a1, a2, m = find_discrete_triangular(mu, cv, skew, a_guess=int(mu * 0.8), b_guess=int(mu * 1.2))

            # Generate one realization from the discrete triangular distribution
            triangular_rv = discrete_triangular_rv(m, a1, a2)
            processing_time = triangular_rv.rvs(size=1)[0] #processing time should be discrete --> need to check

            # ---- ADD RELEASE TIME ----
            # Sample mean release time
            release_mean = np.random.randint(1 * mu, 10 * mu + 1)

            # Sample half-width
            if np.random.rand() < 0.5:
                release_halfwidth = np.random.randint(40, 61)
            else:
                release_halfwidth = np.random.randint(120, 161)

            # Sample final release time
            release_time = np.random.randint(release_mean - release_halfwidth, release_mean + release_halfwidth + 1)

            # ---- ADD DUE DATE ----
            
            #modificata perchè da implementazione descritta sul paper il release time supera la due date
            
            if np.random.rand() < 0.5:
                due_date = np.random.randint(150, 201) + release_time 
            else:
                due_date = np.random.randint(0, 51) + release_time

            # Store job data
            job_data.append({
                "mu": mu,
                "CV": round(cv,3),
                "skew": skew,
                "processing_time": int(processing_time),
                "release_mean": release_mean,
                "release_halfwidth": release_halfwidth,
                "release_time": release_time,
                "due_date": due_date  # New deterministic due date
            })

        # Store instance data
        instances.append({
            "num_jobs": int(num_jobs),
            "jobs": job_data
        })

    return instances

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
