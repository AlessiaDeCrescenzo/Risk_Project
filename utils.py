import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.stats import rv_discrete, triang


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
    res = least_squares(equations, [a_guess, b_guess], args=(mu, cv, gamma1), bounds=(0, np.inf))
    a, b = res.x
    
    c = 3 * mu - a - b
    a, c, b = sorted([round(a), round(b), round(c)])
    
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
    Create a discrete triangular distribution with parameters a, b, c possibly non-integers.

    Parameters:
    - a (float ≥ 0): distance from mode c to lower bound
    - b (float ≥ 0): distance from mode c to upper bound
    - c (float): mode (can be fractional)

    Returns:
    - rv_discrete frozen distribution with integer support
    """
    if a < 0 or b < 0:
        raise ValueError("a and b must be non-negative")

    # integer support covering full range
    support = np.arange(int(np.floor(a)), int(np.ceil(b)) + 1)

    # mode parameter relative to [low, high] in [0,1]
    c_rel = (c - a) / (b - a) if b > a else 0.5  # avoid zero division

    # continuous triangular distribution on [low, high] with mode c_rel
    cont_tri = triang(c_rel, loc=a, scale=b-a)

    # calculate discrete pmf by integrating continuous PDF over integer bins
    pmf = np.array([
        cont_tri.cdf(x + 0.5) - cont_tri.cdf(x - 0.5) for x in support
    ])

    pmf /= pmf.sum()  # normalize to 1 (fix numeric errors)

    return rv_discrete(name="frac_discrete_triangular", values=(support, pmf))

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

def convolve_cdf_with_pmf(F_i, f_j):
    f_i = np.diff(F_i, prepend=0)
    f_conv = np.convolve(f_i, f_j)[:len(F_i)]
    F_conv = np.cumsum(f_conv)
    return np.clip(F_conv, 0, 1)

def compute_max_lateness_cdf(jobs, schedule=None, machine="first", grid_size=3000):
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
    jobs = jobs
    grid = np.arange(grid_size)
    F_Lj_list = []

    # Default schedule if none is provided
    if schedule is None:
        print("It's not possible to compute the lateness because of missing data: a complete schedule needs to be provided")

    # Initialize: first job starts at 0
    first_job_idx = schedule[0]
    first_job = jobs[first_job_idx]
    pt_rv = discrete_triangular_rv_given_param(first_job["a"], first_job["b"], first_job["c"])
    f_i = pmf_from_rv(pt_rv, grid_size)
    F_i = compute_cdf(f_i)
    
    # Release time
    
    if machine == "first":
        r_lo = first_job["rda"] 
        r_hi = first_job["rdb"]
        f_ri = np.zeros(grid_size)
        if r_hi >= r_lo:
            r_vals = np.arange(r_lo, r_hi + 1)
            r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
            for val, p in zip(r_vals, r_probs):
                if 0 <= val < grid_size:
                    f_ri[val] = p
        F_ri = compute_cdf(f_ri)


    # Completion time
    F_Ci = convolve_cdf_with_pmf(F_ri, f_i)

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
        if machine=="first":
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
        F_cj = convolve_cdf_with_pmf(F_sj, f_j)

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


def compute_lower_bound_max_lateness_cdf(jobs, scheduled_jobs, machine="first", grid_size=3000):
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
    jobs = jobs
    num_jobs = len(jobs)
    grid = np.arange(grid_size)
    F_Lj_list = []

    # --- Scheduled jobs first ---
    if scheduled_jobs:
        # Initialize F_Ci: CDF of completion time of the last scheduled job
        first_job_idx = scheduled_jobs[0]
        first_job = jobs[first_job_idx]
        pt_rv = discrete_triangular_rv_given_param(first_job["a"], first_job["b"], first_job["c"])
        f_i = pmf_from_rv(pt_rv, grid_size)
        F_i = compute_cdf(f_i)
        
        # Release time
        
        if machine == "first":
            r_lo = first_job["rda"] 
            r_hi = first_job["rdb"]
            f_ri = np.zeros(grid_size)
            if r_hi >= r_lo:
                r_vals = np.arange(r_lo, r_hi + 1)
                r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
                for val, p in zip(r_vals, r_probs):
                    if 0 <= val < grid_size:
                        f_ri[val] = p
            F_ri = compute_cdf(f_ri)


        # Completion time
        F_Ci = convolve_cdf_with_pmf(F_ri, f_i)

        for idx in scheduled_jobs[1:]:
            job = jobs[idx]

            # Processing time
            pt_rv = discrete_triangular_rv_given_param(job["a"],job["b"], job["c"])
            f_j = pmf_from_rv(pt_rv, grid_size)
            F_j = compute_cdf(f_j)

            # Release time
            
            if machine == "first":
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
            F_cj = convolve_cdf_with_pmf(F_sj, f_j)

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
        #F_Ci[0] = 1.0

    # --- Unscheduled jobs ---
    unscheduled_jobs = [i for i in range(num_jobs) if i not in scheduled_jobs]

    for idx in unscheduled_jobs:
        job = jobs[idx]

        # Processing time
        pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
        f_j = pmf_from_rv(pt_rv, grid_size)

        # Release time
        
        if machine == "first":
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
        F_cj_LB = convolve_cdf_with_pmf(F_sj_LB, f_j)

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

def compute_upper_bound_max_lateness_cdf(jobs, scheduled_jobs, machine = "first", grid_size=3000):
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
    jobs = jobs
    num_jobs = len(jobs)
    grid = np.arange(grid_size)
    F_Lj_list = []

    # --- Scheduled jobs first ---
    if scheduled_jobs:
        # Initialize F_Ci: CDF of completion time of the last scheduled job
        first_job_idx = scheduled_jobs[0]
        first_job = jobs[first_job_idx]
        pt_rv = discrete_triangular_rv_given_param(first_job["a"], first_job["b"], first_job["c"])
        f_i = pmf_from_rv(pt_rv, grid_size)
        F_i = compute_cdf(f_i)
        
        #release dates
        
        if machine == "first":
            r_lo = first_job["rda"] 
            r_hi = first_job["rdb"]
            f_ri = np.zeros(grid_size)
            if r_hi >= r_lo:
                r_vals = np.arange(r_lo, r_hi + 1)
                r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
                for val, p in zip(r_vals, r_probs):
                    if 0 <= val < grid_size:
                        f_ri[val] = p
            F_ri = compute_cdf(f_ri)


        # Completion time
        F_Ci = convolve_cdf_with_pmf(F_ri, f_i)

        for idx in scheduled_jobs[1:]:
            job = jobs[idx]

            # Processing time
            pt_rv = discrete_triangular_rv_given_param(job["a"],job["b"], job["c"])
            f_j = pmf_from_rv(pt_rv, grid_size)
            F_j = compute_cdf(f_j)

            # Release time
            
            if machine == "first":
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
            F_cj = convolve_cdf_with_pmf(F_sj, f_j)

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
    
    if machine == "first":
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
                
                if k==0: 
                    F_j = compute_cdf(f_j)
                    F_C_AminusSminusj=F_j
                    k+=1
                else:
                    F_C_AminusSminusj=convolve_cdf_with_pmf(F_C_AminusSminusj,f_j)
                    k+=1
                    
        f_C_AminusSminusj = np.diff(F_C_AminusSminusj, prepend=0)
        
        # Start time lower bound = product
        F_C_Aminusj_UB =  convolve_cdf_with_pmf((F_Ci*F_rmax),f_C_AminusSminusj)
        
        pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
        f_j = pmf_from_rv(pt_rv, grid_size)
        F_j = compute_cdf(f_j)

        # Completion time lower bound
        F_cj_UB = convolve_cdf_with_pmf(F_C_Aminusj_UB, f_j)

        # Lateness distribution
        d_j = job["dd"]
        F_lj_UB = np.ones(grid_size)
        if d_j < grid_size:
            F_lj_UB[:grid_size - d_j] = F_cj_UB[d_j:]

        F_Lj_list.append(F_lj_UB)
        

    # --- Final step: maximum lateness CDF is the product of individual lateness CDFs ---
    F_Lmax_upper_bound = np.ones(grid_size)
    for F_lj in F_Lj_list:
        F_Lmax_upper_bound *= F_lj

    return grid, F_Lmax_upper_bound


