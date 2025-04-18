import numpy as np
from scipy.optimize import fsolve
from scipy.stats import rv_discrete
from scipy.signal import convolve


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
    values = np.arange(m - a1, m + a2 + 1)  # Support of the distribution

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

def compute_max_lateness_cdf(instance, grid_size=3000):
    jobs = instance["jobs"]
    num_jobs = instance["num_jobs"]

    grid = np.arange(grid_size)
    F_Lj_list = []

    # First job starts at time 0, its completion time is its processing time
    first_job = jobs[0]
    a1, a2, m = find_discrete_triangular(first_job["mu"], first_job["CV"], first_job["skew"])
    pt_rv = discrete_triangular_rv(m, a1, a2)
    f_ci = pmf_from_rv(pt_rv, grid_size)
    F_Ci = compute_cdf(f_ci)

    for j in range(1, num_jobs):
        job = jobs[j]

        # Processing time
        a1, a2, m = find_discrete_triangular(job["mu"], job["CV"], job["skew"])
        pt_rv = discrete_triangular_rv(m, a1, a2)
        f_j = pmf_from_rv(pt_rv, grid_size)
        F_j = compute_cdf(f_j)

        # Release time: uniform distribution
        r_lo = job["release_mean"] - job["release_halfwidth"]
        r_hi = job["release_mean"] + job["release_halfwidth"]
        r_vals = np.arange(r_lo, r_hi + 1)
        r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
        f_rj = np.zeros(grid_size)
        for val, p in zip(r_vals, r_probs):
            if 0 <= val < grid_size:
                f_rj[val] = p
        F_rj = compute_cdf(f_rj)

        # Start time: F_s_j = F_c_i * F_rj
        F_sj = F_Ci[:grid_size] * F_rj[:grid_size]

        # Completion time: F_c_j = F_s_j * f_j
        F_cj = convolve(F_sj, f_j)[:grid_size]

        # Lateness: F_L_j(t) = F_C_j(t + d_j)
        d_j = job["due_date"]
        shift = d_j
        F_lj = np.zeros(grid_size)
        if shift < grid_size:
            F_lj[:grid_size - shift] = F_cj[shift:]

        F_Lj_list.append(F_lj)

    # Maximum lateness CDF
    F_Lmax = np.ones(grid_size)
    for F_lj in F_Lj_list:
        F_Lmax *= F_lj

    return grid, F_Lmax

def compute_lower_bound_max_lateness(instance, scheduled_indices, grid_size=3000):
    jobs = instance["jobs"]
    num_jobs = instance["num_jobs"]
    all_indices = set(range(num_jobs))
    unscheduled_indices = list(all_indices - set(scheduled_indices))
    
    grid = np.arange(grid_size)
    F_Lj_list = []
    
    # ---- Step 1: Compute F_C_S (completion time CDF of scheduled jobs) ----
    F_Cs = np.zeros(grid_size)
    F_Cs[0] = 1.0  # identity for convolution
    
    for j in scheduled_indices:
        job = jobs[j]
        a1, a2, m = find_discrete_triangular(job["mu"], job["CV"], job["skew"])
        pt_rv = discrete_triangular_rv(m, a1, a2)
        f_j = pmf_from_rv(pt_rv, grid_size)
        F_Cs = convolve(F_Cs, f_j)[:grid_size]
    
    # Save F_Cs as it is used again for unscheduled jobs' lower bounds
    F_C_S = compute_cdf(F_Cs)
    
    # ---- Step 2: Compute F_L_j for scheduled jobs ----
    for j in scheduled_indices:
        job = jobs[j]
        a1, a2, m = find_discrete_triangular(job["mu"], job["CV"], job["skew"])
        pt_rv = discrete_triangular_rv(m, a1, a2)
        f_j = pmf_from_rv(pt_rv, grid_size)
        
        # Release time
        r_lo = job["release_mean"] - job["release_halfwidth"]
        r_hi = job["release_mean"] + job["release_halfwidth"]
        r_vals = np.arange(r_lo, r_hi + 1)
        r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
        f_rj = np.zeros(grid_size)
        for val, p in zip(r_vals, r_probs):
            if 0 <= val < grid_size:
                f_rj[val] = p
        F_rj = compute_cdf(f_rj)
        
        # Assume job starts as soon as it can: max(release_time, completion of previous)
        F_sj = F_C_S * F_rj
        F_cj = convolve(F_sj, f_j)[:grid_size]
        
        # Lateness
        d_j = job["due_date"]
        F_lj = np.zeros(grid_size)
        if d_j < grid_size:
            F_lj[:grid_size - d_j] = F_cj[d_j:]
        F_Lj_list.append(F_lj)
    
    # ---- Step 3: Lower Bound for F_L_j for unscheduled jobs ----
    for j in unscheduled_indices:
        job = jobs[j]
        a1, a2, m = find_discrete_triangular(job["mu"], job["CV"], job["skew"])
        pt_rv = discrete_triangular_rv(m, a1, a2)
        f_j = pmf_from_rv(pt_rv, grid_size)
        
        # Release time as uniform
        r_lo = job["release_mean"] - job["release_halfwidth"]
        r_hi = job["release_mean"] + job["release_halfwidth"]
        r_vals = np.arange(r_lo, r_hi + 1)
        r_probs = np.full_like(r_vals, 1 / len(r_vals), dtype=float)
        f_rj = np.zeros(grid_size)
        for val, p in zip(r_vals, r_probs):
            if 0 <= val < grid_size:
                f_rj[val] = p
        F_rj = compute_cdf(f_rj)
        
        # Lower bound start time CDF
        F_sj_lb = F_C_S * F_rj
        F_cj_lb = convolve(F_sj_lb, f_j)[:grid_size]
        
        # Lateness LB
        d_j = job["due_date"]
        F_lj_lb = np.zeros(grid_size)
        if d_j < grid_size:
            F_lj_lb[:grid_size - d_j] = F_cj_lb[d_j:]
        F_Lj_list.append(F_lj_lb)
    
    # ---- Step 4: Final Lower Bound for Max Lateness ----
    F_Lmax_LB = np.ones(grid_size)
    for F_lj in F_Lj_list:
        F_Lmax_LB *= F_lj
    
    return grid, F_Lmax_LB
