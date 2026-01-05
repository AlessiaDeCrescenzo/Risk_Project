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
        num_jobs = np.random.choice([5, 10])  # Randomly pick number of jobs
        
        job_first_op = []  # Store data for each job
        job_second_op = []
        for _ in range(num_jobs):
            # Sample mu from the specified distributions
            if np.random.rand() < 0.5:
                mu_1 = np.random.randint(50, 101)
                mu_2 = np.random.randint(50, 101)
            else:
                mu_1 = np.random.randint(25, 76)
                mu_2 = np.random.randint(25, 76)

            # Sample CV from the specified distributions
            if np.random.rand() < 0.5:
                cv_1 = np.random.uniform(1.4, 1.6)
                cv_2 = np.random.uniform(1.4, 1.6)
            else:
                cv_1 = np.random.uniform(0.4, 0.6)
                cv_2 = np.random.uniform(0.4, 0.6)

            # Randomly assign skewness from the given values
            skew = np.random.choice([-0.5, 0, 0.5])

            # Solve for a1, a2, m (mode) using the existing method
            a1, c1, b1 = find_discrete_triangular(mu_1, cv_1, skew, a_guess=int(mu_1 * 0.8), b_guess=int(mu_1 * 1.2))
            a2, c2, b2=find_discrete_triangular(mu_2, cv_2, skew, a_guess=int(mu_2 * 0.8), b_guess=int(mu_2 * 1.2))

            # add release time distribution
            
            # Sample mean release time
            release_mean = np.random.randint(mu_1, 10 * mu_1 + 1)

            # Sample half-width
            if np.random.rand() < 0.5:
                release_halfwidth = np.random.randint(40, 61)
            else:
                release_halfwidth = np.random.randint(120, 161)

            
            rda= max(release_mean -release_halfwidth,0)
            rdb=release_mean + release_halfwidth + 1

           #add due date
            
            if np.random.rand() < 0.5:
                due_date = np.random.randint(150, 201) + rda
            else:
                due_date = np.random.randint(0, 51) + rdb

            # Store job data
            job_first_op.append({
                "a": a1,
                "b": b1,
                "c": c1,
                "rda": rda,
                "rdb": rdb,
                "dd": due_date  # New deterministic due date
            })
            
            job_second_op.append({
                "a": a2,
                "b": b2,
                "c": c2,
                "rda": rda,
                "rdb": rdb,
                "dd": due_date
            })

        # Store instance data
        instances.append({
            "num_jobs": int(num_jobs),
            "jobs1": job_first_op,
            "jobs2": job_second_op
        })

    return instances

