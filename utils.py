import numpy as np
from scipy.optimize import fsolve
from scipy.stats import rv_discrete
from scipy.signal import convolve,fftconvolve
import matplotlib.pyplot as plt
from itertools import permutations
import time

import json
import re

import re
import json

def process_file(file_path):
    instances = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Read the first row for number of jobs per instance
        num_jobs_per_instance = int(lines[0].strip())

        # Read the 4th row (index 3 because Python is zero-based)
        row = lines[3].strip()

        if row.startswith('{') and row.endswith('}'):
            row = row[1:-1]  # remove outermost braces
        
        # Use a regex to find top-level sets of braces
        pattern = re.compile(r'\{((?:[^{}]|\{[^{}]*\})*)\}')
        instances_raw = pattern.findall(row)

        # Now parse each instance
        parsed_instances = []
        for instance in instances_raw:
            tuples = re.findall(r'\{([^{}]*)\}', instance)
            for t in tuples:
                numbers = [float(x) for x in t.split(',')]
                parsed_instances.append(tuple(numbers))

        # Now parsed_instances is a flat list of tuples (each with 6 values)
        # Divide them into instances with num_jobs_per_instance each
        num_total_jobs = len(parsed_instances)
        num_instances = num_total_jobs // num_jobs_per_instance

        for i in range(num_instances):
            jobs = []
            start_index = i * num_jobs_per_instance
            end_index = start_index + num_jobs_per_instance

            for j in range(start_index, end_index):
                job_tuple = parsed_instances[j]
                job = {
                    "a": job_tuple[0],
                    "b": job_tuple[1],
                    "c": job_tuple[2],
                    "rda": int(job_tuple[3]),
                    "rdb": int(job_tuple[4]),
                    "dd": int(job_tuple[5])
                }
                jobs.append(job)

            instances.append({
                "num_jobs": num_jobs_per_instance,
                "jobs": jobs
            })

    # Save to a JSON file
    with open("instances2.json", "w") as out_file:
        json.dump(instances, out_file, indent=4)

    print(f"Processed {len(instances)} instances and saved to 'instances2.json'.")
    
    return instances



def equations(vars, mu, cv, gamma1):
    a, b = vars
    c = 3 * mu - a - b  # Express c in terms of a and b
    
    # Compute variance
    var = (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18
    sigma = np.sqrt(var)
    
    # Compute coefficient of variation
    cv_calc = sigma / mu
    
    # Compute skewness
    num = np.sqrt(2) * (a + b - 2*c) * (2*a - b - c) * (a - 2*b + c)
    denom = 5 * (var ** 1.5)
    gamma1_calc = num / denom
    
    return [cv_calc - cv, gamma1_calc - gamma1]

def find_discrete_triangular(mu, cv, gamma1, a_guess=10, b_guess=20):
    solution = fsolve(equations, (a_guess, b_guess), args=(mu, cv, gamma1))
    a, b = solution
    c = 3 * mu - a - b
    
    #Ensure integer values and correct ordering
    a, b, c = sorted([round(a), round(b), round(c)])
    
    return a, c, b


def discrete_triangular_rv(mean, cv, gamma1, a1_guess= 10, a2_guess= 20):
    """Creates an asymmetric discrete triangular distribution using the correct PMF."""
    a1,m, a2 = find_discrete_triangular(mean, cv, gamma1, a1_guess, a2_guess)
    a,b= sorted([m - a1, m + a2 + 1])
    values = np.arange(a,b)  # Support of the distribution

    # Compute PMF using the exact formula from the definition
    normalization_factor = (a1 + a2 + 2) / 2
    probs = np.array([
        (1 - (m - y) / (a1 + 1)) / normalization_factor if y < m else
        (1 - (y - m) / (a2 + 1)) / normalization_factor
        for y in values
    ])
    
    probs /= np.sum(probs)  # Normalize to ensure the total probability is 1

    # Safety check
    assert np.isclose(np.sum(probs), 1), "PMF probabilities do not sum to 1!"
    
    return rv_discrete(name='asym_discrete_triangular', values=(values, probs))

def discrete_triangular_rv_given_param(a, b, c):
    """
    Creates an asymmetric discrete triangular distribution using a non-negative, normalized PMF.

    Parameters:
    - a: number of steps from mode c to the lower bound (c - a)
    - b: number of steps from mode c to the upper bound (c + b)
    - c: mode (integer)

    Returns:
    - rv_discrete instance
    """
    # Support values
    support = np.arange(c - a, c + b + 1)

    # Build unnormalized triangular probabilities
    probs = []
    for x in support:
        if x <= c:
            prob = (x - (c - a) + 1)  # ascending from left to mode
        else:
            prob = ((c + b) - x + 1)  # descending from mode to right
        probs.append(prob)

    probs = np.array(probs, dtype=float)
    probs /= probs.sum()  # Normalize to sum to 1

    # Safety checks
    assert np.all(probs >= 0), "Negative probabilities found!"
    assert np.isclose(probs.sum(), 1.0), "Probabilities do not sum to 1"

    return rv_discrete(name='asym_discrete_triangular', values=(support, probs))

def pmf_from_rv(rv, max_len=1000):
    x = rv.xk.astype(int)
    probs = rv.pk
    pmf = np.zeros(max_len)
    for val, p in zip(x, probs):
        if 0 <= val < max_len:
            pmf[val] += p
    return pmf

def compute_cdf(pmf):
    return np.cumsum(pmf)

def convolve_cdf_with_cdf(F_i, f_j):
    grid_size = len(F_i)
    F_conv = np.zeros(grid_size)
    
    for y, prob in enumerate(f_j):
        if prob == 0:
            continue
        shift = np.arange(grid_size) - y
        shifted_F = np.where(shift >= 0, F_i[shift], 0)
        F_conv += prob * shifted_F
    
    F_conv = np.clip(F_conv, 0, 1)
    return F_conv

def compute_max_lateness_cdf(instance, schedule=None, grid_size=3000):
    """
    Compute the CDF of the maximum lateness, following a given schedule.

    Args:
        instance (dict): An instance containing job information.
        schedule (list or None): List of job indices indicating the processing order.
                                 If None, jobs are scheduled by their natural order (0,1,2,...).
        grid_size (int): Time grid size for discretization.

    Returns:
        grid (np.ndarray): Time points.
        F_Lmax (np.ndarray): CDF of maximum lateness.
    """
    jobs = instance["jobs"]
    num_jobs = instance["num_jobs"]
    grid = np.arange(grid_size)
    F_Lj_list = []

    # Default schedule if none is provided
    if schedule is None:
        schedule = list(range(num_jobs))

    # Initialize: first job starts at 0
    first_job_idx = schedule[0]
    first_job = jobs[first_job_idx]
    pt_rv = discrete_triangular_rv_given_param(first_job["a"], first_job["b"], first_job["c"])
    f_ci = pmf_from_rv(pt_rv, grid_size)
    F_Ci = compute_cdf(f_ci)

    # Completion time of the first job is directly its processing time
    d_j = first_job["dd"]
    F_lj = np.ones(grid_size)
    if d_j < grid_size:
        F_lj[:grid_size - d_j] = F_Ci[d_j:]

    F_Lj_list.append(F_lj)

    # Process remaining jobs
    for idx in schedule[1:]:
        job = jobs[idx]

        # Processing time
        pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
        f_j = pmf_from_rv(pt_rv, grid_size)
        F_j = compute_cdf(f_j)

        # Release time
        r_lo = job["rda"] 
        r_hi = job["rdb"]
        f_rj = np.zeros(grid_size)
        if r_hi >= r_lo:
            r_vals = np.arange(r_lo, r_hi + 1)
            r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
            for val, p in zip(r_vals, r_probs):
                if 0 <= val < grid_size:
                    f_rj[val] = p
        F_rj = compute_cdf(f_rj)

        # Start time: maximum between previous completion and release time (lower bound by product)
        F_sj = F_Ci * F_rj

        # Completion time
        F_cj = convolve_cdf_with_cdf(F_sj, f_j)

        # Lateness
        d_j = job["dd"]
        F_lj = np.ones(grid_size)
        if d_j < grid_size:
            F_lj[:grid_size - d_j] = F_cj[d_j:]

        F_Lj_list.append(F_lj)

        # Update F_Ci for the next job
        F_Ci = F_cj

    # Maximum lateness CDF: product of all lateness CDFs
    F_Lmax = np.ones(grid_size)
    for F_lj in F_Lj_list:
        F_Lmax *= F_lj

    return grid, F_Lmax


def compute_lower_bound_max_lateness_cdf(instance, scheduled_jobs, grid_size=3000):
    """
    Compute the lower bound of the maximum lateness CDF given a partial schedule.
    All operations are correctly defined in terms of CDFs.

    Args:
        instance (dict): An instance containing job information.
        scheduled_jobs (list): List of job indices already scheduled, in order.
        grid_size (int): Time grid size for discretization.

    Returns:
        grid (np.ndarray): Time points.
        F_Lmax_lower_bound (np.ndarray): Lower bound CDF of maximum lateness.
    """
    jobs = instance["jobs"]
    num_jobs = instance["num_jobs"]
    grid = np.arange(grid_size)
    F_Lj_list = []

    # --- Scheduled jobs first ---
    if scheduled_jobs:
        # Initialize F_Ci: CDF of completion time of the last scheduled job
        first_job_idx = scheduled_jobs[0]
        first_job = jobs[first_job_idx]
        pt_rv = discrete_triangular_rv_given_param(first_job["a"], first_job["b"], first_job["c"])
        f_ci = pmf_from_rv(pt_rv, grid_size)
        F_Ci = compute_cdf(f_ci)

        for idx in scheduled_jobs:
            job = jobs[idx]

            # Processing time
            pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
            f_j = pmf_from_rv(pt_rv, grid_size)
            F_j = compute_cdf(f_j)

            # Release time
            r_lo = job["rda"]
            r_hi = job["rdb"]
            f_rj = np.zeros(grid_size)
            if r_hi >= r_lo:  # just to be safe
                r_vals = np.arange(r_lo, r_hi + 1)
                r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
                for val, p in zip(r_vals, r_probs):
                    if 0 <= val < grid_size:
                        f_rj[val] = p
            F_rj = compute_cdf(f_rj)

            # Start time distribution = max(C_i, r_j) -> lower bound as product
            F_sj = F_Ci * F_rj

            # Completion time distribution
            F_cj = convolve_cdf_with_cdf(F_sj, f_j)

            # Lateness distribution
            d_j = job["dd"]
            F_lj = np.ones(grid_size)
            if d_j < grid_size:
                F_lj[:grid_size - d_j] = F_cj[d_j:]

            F_Lj_list.append(F_lj)

            # Update F_Ci
            F_Ci = F_cj

    else:
        F_Ci = np.ones(grid_size)

    # --- Unscheduled jobs ---
    unscheduled_jobs = [i for i in range(num_jobs) if i not in scheduled_jobs]

    for idx in unscheduled_jobs:
        job = jobs[idx]

        # Processing time
        pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
        f_j = pmf_from_rv(pt_rv, grid_size)
        F_j = compute_cdf(f_j)

        # Release time
        r_lo = job["rda"] 
        r_hi = job["rdb"]
        f_rj = np.zeros(grid_size)
        if r_hi >= r_lo:
            r_vals = np.arange(r_lo, r_hi + 1)
            r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
            for val, p in zip(r_vals, r_probs):
                if 0 <= val < grid_size:
                    f_rj[val] = p
        F_rj = compute_cdf(f_rj)

        # Start time lower bound = product
        F_sj_LB = F_Ci * F_rj

        # Completion time lower bound
        F_cj_LB = convolve_cdf_with_cdf(F_sj_LB, f_j)

        # Lateness distribution
        d_j = job["dd"]
        F_lj_LB = np.ones(grid_size)
        if d_j < grid_size:
            F_lj_LB[:grid_size - d_j] = F_cj_LB[d_j:]

        F_Lj_list.append(F_lj_LB)

    # --- Final step: maximum lateness CDF is the product of individual lateness CDFs ---
    F_Lmax_lower_bound = np.ones(grid_size)
    for F_lj in F_Lj_list:
        F_Lmax_lower_bound *= F_lj
        
    return grid, F_Lmax_lower_bound

def compute_upper_bound_max_lateness_cdf(instance, scheduled_jobs, grid_size=3000):
    """
    Compute the upper bound of the maximum lateness CDF given a partial schedule.
    All operations are correctly defined in terms of CDFs.

    Args:
        instance (dict): An instance containing job information.
        scheduled_jobs (list): List of job indices already scheduled, in order.
        grid_size (int): Time grid size for discretization.

    Returns:
        grid (np.ndarray): Time points.
        F_Lmax_lower_bound (np.ndarray): Lower bound CDF of maximum lateness.
    """
    jobs = instance["jobs"]
    num_jobs = instance["num_jobs"]
    grid = np.arange(grid_size)
    F_Lj_list = []

    # --- Scheduled jobs first ---
    if scheduled_jobs:
        # Initialize F_Ci: CDF of completion time of the last scheduled job
        first_job_idx = scheduled_jobs[0]
        first_job = jobs[first_job_idx]
        pt_rv = discrete_triangular_rv_given_param(first_job["a"], first_job["b"], first_job["c"])
        f_ci = pmf_from_rv(pt_rv, grid_size)
        F_Ci = compute_cdf(f_ci)

        for idx in scheduled_jobs:
            job = jobs[idx]

            # Processing time
            pt_rv = discrete_triangular_rv_given_param(job["a"],job["b"], job["c"])
            f_j = pmf_from_rv(pt_rv, grid_size)
            F_j = compute_cdf(f_j)

            # Release time
            r_lo = job["rda"]
            r_hi = job["rdb"] 
            f_rj = np.zeros(grid_size)
            if r_hi >= r_lo:  # just to be safe
                r_vals = np.arange(r_lo, r_hi + 1)
                r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
                for val, p in zip(r_vals, r_probs):
                    if 0 <= val < grid_size:
                        f_rj[val] = p
            F_rj = compute_cdf(f_rj)

            # Start time distribution = max(C_i, r_j) -> lower bound as product
            F_sj = F_Ci * F_rj

            # Completion time distribution
            F_cj = convolve_cdf_with_cdf(F_sj, f_j)

            # Lateness distribution
            d_j = job["dd"]
            F_lj = np.ones(grid_size)
            if d_j < grid_size:
                F_lj[:grid_size - d_j] = F_cj[d_j:]

            F_Lj_list.append(F_lj)

            # Update F_Ci
            F_Ci = F_cj

    else:
    # No scheduled jobs: assume "dummy" job completed at time 0
        F_Ci = np.zeros(grid_size)
        F_Ci[0] = 1.0

    # --- Unscheduled jobs ---
    unscheduled_jobs = [i for i in range(num_jobs) if i not in scheduled_jobs]
    
    job=jobs[unscheduled_jobs[0]]
    r_lo = job["rda"]
    r_hi = job["rdb"]
    f_rj = np.zeros(grid_size)
    if r_hi >= r_lo:
        r_vals = np.arange(r_lo, r_hi + 1)
        r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
        for val, p in zip(r_vals, r_probs):
            if 0 <= val < grid_size:
                f_rj[val] = p
    F_rmax = compute_cdf(f_rj)

    for idx in unscheduled_jobs[1:]:
        job = jobs[idx]

        # Processing time of r_max, i.e, the product of the cdf of all release times in unscheduled jobs
        r_lo = job["rda"]
        r_hi = job["rdb"] 
        f_rj = np.zeros(grid_size)
        if r_hi >= r_lo:
            r_vals = np.arange(r_lo, r_hi + 1)
            r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
            for val, p in zip(r_vals, r_probs):
                if 0 <= val < grid_size:
                    f_rj[val] = p
        F_rj = compute_cdf(f_rj)
        F_rmax*=F_rj
        
    
    for idx in unscheduled_jobs:
        F_C_AminusSminusj = np.ones(grid_size)
        k=0
        
        job = jobs[idx]
        
        for i in unscheduled_jobs:
            if i != idx: 
                job_i=jobs[i]
                pt_rv = discrete_triangular_rv_given_param(job_i["a"], job_i["b"], job_i["c"])
                f_j = pmf_from_rv(pt_rv, grid_size)
                F_j = compute_cdf(f_j)
                
                if k==0: 
                    F_C_AminusSminusj=F_j
                    k+=1
                else:
                    F_C_AminusSminusj*=F_j
                    k+=1
                    
        f_C_AminusSminusj = np.diff(F_C_AminusSminusj, prepend=0)
        
        # Start time lower bound = product
        F_C_Aminusj_UB =  convolve_cdf_with_cdf((F_Ci*F_rmax),f_C_AminusSminusj)
        
        pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
        f_j = pmf_from_rv(pt_rv, grid_size)
        F_j = compute_cdf(f_j)

        # Completion time lower bound
        F_cj_LB = convolve_cdf_with_cdf(F_C_Aminusj_UB, f_j)

        # Lateness distribution
        d_j = job["dd"]
        F_lj_LB = np.ones(grid_size)
        if d_j < grid_size:
            F_lj_LB[:grid_size - d_j] = F_cj_LB[d_j:]

        F_Lj_list.append(F_lj_LB)
        

    # --- Final step: maximum lateness CDF is the product of individual lateness CDFs ---
    F_Lmax_upper_bound = np.ones(grid_size)
    for F_lj in F_Lj_list:
        F_Lmax_upper_bound *= F_lj

    return grid, F_Lmax_upper_bound

def build_schedule_with_bounds(instance, threshold=0.975):

    num_jobs = instance['num_jobs']
    pending_nodes_by_level = {0: [(0, [], list(range(num_jobs)))]}
    best_schedule = None
    best_ub = float('inf')
    
    visited_nodes_count = 0
    visited_levels = set()
    best_partial_schedules = {}
    current_level = 0
    ub_updated_level = 0  # New: track where the best UB was last updated
    start_time = time.time()

    while current_level in pending_nodes_by_level and pending_nodes_by_level[current_level]:
        nodes_at_level = pending_nodes_by_level[current_level]

        # Determine which nodes to consider
        if current_level not in visited_levels and current_level != 0:
            visited_levels.add(current_level)
            considered_nodes = [(best_node_data[2], best_node_data[0], best_node_data[3])]
        else:
            best_sched = best_partial_schedules.get(current_level)
            if best_sched is not None:
                considered_nodes = [n for n in nodes_at_level if n[1] == best_sched]
            else:
                considered_nodes = nodes_at_level

        children = []
        best_node_data = None  # (schedule, ub, lb, remaining_jobs)

        for lb, sched, rem in considered_nodes:
            if len(rem) <= 2:
                visited_nodes_count += 1
                best_local_lateness = float('inf')
                best_local_schedule = None

                for perm in permutations(rem):
                    final_schedule = sched + list(perm)
                    grid, F = compute_max_lateness_cdf(instance, final_schedule)
                    idx = np.where(F >= threshold)[0]
                    lateness = grid[idx[0]] if len(idx) > 0 else float('inf')

                    if lateness < best_local_lateness:
                        best_local_lateness = lateness
                        best_local_schedule = final_schedule

                # Always update best_ub and best_schedule from this level
                best_ub = best_local_lateness
                best_schedule = best_local_schedule
                ub_updated_level = current_level
                print(f"New global best UB {best_ub:.3f} with complete schedule {best_schedule}")

                continue


            for job in rem:
                visited_nodes_count += 1
                new_sched = sched + [job]
                new_rem = [j for j in rem if j != job]

                grid_lb, F_lb = compute_lower_bound_max_lateness_cdf(instance, new_sched)
                grid_ub, F_ub = compute_upper_bound_max_lateness_cdf(instance, new_sched)

                idx_lb = np.where(F_lb >= threshold)[0]
                lb_val = grid_lb[idx_lb[0]] if len(idx_lb) > 0 else float('inf')

                idx_ub = np.where(F_ub >= threshold)[0]
                ub_val = grid_ub[idx_ub[0]] if len(idx_ub) > 0 else float('inf')

                if (best_node_data is None) or (ub_val < best_node_data[1]):
                    best_node_data = (new_sched, ub_val, lb_val, new_rem)

                children.append((lb_val, new_sched, new_rem))

        if best_node_data:
            best_ub = best_node_data[1]
            ub_updated_level = current_level  # Track level where best UB updated
            best_partial_schedules[current_level] = best_node_data[0]
            print(f"New global best UB {best_ub:.3f} with partial schedule {best_node_data[0]}")

        # Prune children
        surviving_children = [child for child in children if child[0] <= best_ub]
        pruned_count = len(children) - len(surviving_children)
        if pruned_count > 0:
            print(f"Pruned {pruned_count} children at level {current_level + 1} due to LB >= best_ub")

        if surviving_children:
            pending_nodes_by_level[current_level + 1] = pending_nodes_by_level.get(current_level + 1, []) + surviving_children
        else:
            pending_nodes_by_level[current_level + 1] = []

        current_level += 1

        # Backtrack and explore only from ub_updated_level onward
        for lvl in range(ub_updated_level, current_level):
            if lvl not in pending_nodes_by_level:
                continue

            new_nodes = []
            for node in pending_nodes_by_level[lvl]:
                lb, sched, rem = node
                if lb >= best_ub:
                    print(f"Back-pruned node at level {lvl} with LB={lb:.3f} >= best_ub={best_ub:.3f}")
                    continue

                # Explore this node up to the current depth
                temp_sched = sched
                temp_rem = rem
                temp_level = len(temp_sched)

                while temp_level < current_level and temp_rem:
                    best_local = None
                    for job in temp_rem:
                        visited_nodes_count += 1
                        new_sched = temp_sched + [job]
                        new_rem = [j for j in temp_rem if j != job]

                        grid_ub, F_ub = compute_upper_bound_max_lateness_cdf(instance, new_sched)
                        idx_ub = np.where(F_ub >= threshold)[0]
                        ub_val = grid_ub[idx_ub[0]] if len(idx_ub) > 0 else float('inf')
                        
                        grid_lb, F_lb = compute_lower_bound_max_lateness_cdf(instance, new_sched)
                        idx_lb = np.where(F_lb >= threshold)[0]
                        lb_val = grid_lb[idx_lb[0]] if len(idx_lb) > 0 else float('inf')

                        if best_local is None or ub_val < best_local[1]:
                            best_local = (new_sched, ub_val,lb_val, new_rem)

                    if best_local:
                        temp_sched, ub_val,lb_val, temp_rem = best_local
                        temp_level += 1

                        if ub_val < best_ub:
                            best_ub = ub_val
                            best_partial_schedules[temp_level - 1] = temp_sched
                            ub_updated_level = temp_level - 1
                            best_node_data=best_local
                            print(f"Updated best UB {best_ub:.3f} from backtracked path with partial schedule {temp_sched}")

                new_nodes.append(node)

            pending_nodes_by_level[lvl] = new_nodes

    elapsed_time = time.time() - start_time
    print("\nBest schedule found:", best_schedule)
    print(f"Visited nodes:{visited_nodes_count}")
    print(f"Best UB (lateness): {best_ub:.3f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return best_schedule,elapsed_time,visited_nodes_count
