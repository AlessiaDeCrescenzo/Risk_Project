from utils import *
import numpy as np

class CdF_estimator():
    """Estimator of the CDF of the maximum lateness, given a schedule. 
    It also provides lower and upper bounds for the CDFs of partial schedules.
    
        Attributes:
            jobs (list): List of job dictionaries, each containing parameters a, b, c, rda, rdb, dd.
            num_jobs (int): Number of jobs.
            cdfs_estimate (dict): Dictionary to store estimated CDFs of completion times for fixed partial schedules.
            
    """
    
    def __init__(self, n_jobs, jobs):
        self.jobs=jobs
        self.num_jobs=n_jobs
        self.cdfs_estimate={}
    
    
    def compute_max_lateness_cdf(self, schedule=None, n_machines=1,prev=None, grid_size=2500):
        
        """
        Compute the CDF of the maximum lateness, following a given schedule.
        
        Args:
            schedule (list): A list of job indices representing the schedule order.
            first (bool): Whether to consider the first machine's release times (if True) or not (if False). Default set to True.
            prev (int): previous machine in the schedule, if any. Default set to None, as for the first machine.
            grid_size (int): The size of the grid for CDF estimation.
            
        Returns:
            grid (np.ndarray): Time points.
            F_Lmax (np.ndarray): CDF of maximum lateness.
        """
        
        #grid of data points
        grid = np.arange(grid_size)
        
        # List to store the CDF of lateness for each job
        F_Lj_list = []

        # If schedule is None, this method cannot compute the CDF
        if schedule is None:
            print("It's not possible to compute the lateness because of missing data: a complete schedule needs to be provided")

        # for each job in the schedule, compute the CDF of its lateness
        for j in range(0,len(schedule)-2):
                    
            idx=schedule[j]
            job = next((j for j in self.jobs if j['op'] == idx), None)

            d_j = job["dd"]
            F_lj = np.ones(grid_size)
            
            if d_j < grid_size:
                F_lj[:grid_size - d_j] = self.cdfs_estimate[tuple(schedule[:j+1])][n_machines-1][d_j:]

            F_Lj_list.append(F_lj)
            
        for el in range(2,0,-1):    
            
            idx=schedule[-el]
            
            for i in range(n_machines):
                
                op=(idx[0],i)
                job = next((j for j in self.jobs if j['op'] == op), None)
                # Processing time
                pt_rv = discrete_triangular_rv_given_param(job["a"],job["b"], job["c"])
                f_j = pmf_from_rv(pt_rv, grid_size)
                    
                if i==0:
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
                    
                else:
                    
                    F_rj = self.cdfs_estimate[tuple(schedule)][i-1]

                # Start time distribution = max(C_i, r_j) -> lower bound as product
                if len(schedule)>1:       
                    F_sj = self.cdfs_estimate[tuple(schedule[:-el])][i] * F_rj
                else:
                    F_sj = F_rj

                # Completion time distribution
                F_cj = convolve_cdf_with_pmf(F_sj, f_j)

                if i==n_machines-1:
                    # Lateness distribution
                    d_j = job["dd"]
                    F_lj = np.ones(grid_size)
                    if d_j < grid_size:
                        F_lj[:grid_size - d_j] = F_cj[d_j:]

                    F_Lj_list.append(F_lj)

                # # Update F_Ci
                # F_Ci = F_cj
                
                
                if i==0:
                    schedule_key = tuple(schedule[:-(el-1)])
                    self.cdfs_estimate[schedule_key] = [None] * n_machines
                    
                
                self.cdfs_estimate[tuple(schedule[:-(el-1)])][i]=F_cj


        # Maximum lateness CDF: product of all lateness CDFs
        F_Lmax = np.ones(grid_size)
        for F_lj in F_Lj_list:
            F_Lmax *= F_lj

        return grid, F_Lmax
    
    def compute_lower_bound_max_lateness_cdf(self, scheduled_jobs, n_machines=1, grid_size=2500):
        
        """
        Compute the lower bound of the maximum lateness CDF given a partial schedule.
        All operations are correctly defined in terms of CDFs.

        Returns:
            grid (np.ndarray): Time points.
            F_Lmax_lower_bound (np.ndarray): Lower bound CDF of maximum lateness.
        """
        
        grid = np.arange(grid_size)
        F_Lj_list = []

        # --- Scheduled jobs first ---
        if scheduled_jobs:
            
            if len(scheduled_jobs)>1:
                
                for j in range(1,len(scheduled_jobs)):
                    
                    idx=scheduled_jobs[j-1]
                    job = next((j for j in self.jobs if j['op'] == idx), None)

                    d_j = job["dd"]
                    F_lj = np.ones(grid_size)
                    
                    n=len(scheduled_jobs)
                    
                    if d_j < grid_size:
                        F_lj[:grid_size - d_j] = self.cdfs_estimate[tuple(scheduled_jobs[:j])][n_machines-1][d_j:]

                    F_Lj_list.append(F_lj)
                    
            idx=scheduled_jobs[-1]

            for i in range(n_machines):
                
                op=(idx[0],i)
                job = next((j for j in self.jobs if j['op'] == op), None)
                # Processing time
                pt_rv = discrete_triangular_rv_given_param(job["a"],job["b"], job["c"])
                f_j = pmf_from_rv(pt_rv, grid_size)
                    
                if i==0:
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
                    
                else:
                    
                    F_rj = self.cdfs_estimate[tuple(scheduled_jobs)][i-1]

                # Start time distribution = max(C_i, r_j) -> lower bound as product
                if len(scheduled_jobs)>1:       
                    F_sj = self.cdfs_estimate[tuple(scheduled_jobs[:-1])][i] * F_rj
                else:
                    F_sj = F_rj

                # Completion time distribution
                F_cj = convolve_cdf_with_pmf(F_sj, f_j)

                if i==n_machines-1:
                    # Lateness distribution
                    d_j = job["dd"]
                    F_lj = np.ones(grid_size)
                    if d_j < grid_size:
                        F_lj[:grid_size - d_j] = F_cj[d_j:]

                    F_Lj_list.append(F_lj)

                # # Update F_Ci
                # F_Ci = F_cj
                
                
                if i==0:
                    schedule_key = tuple(scheduled_jobs)
                    self.cdfs_estimate[schedule_key] = [None] * n_machines
                    
                
                self.cdfs_estimate[tuple(scheduled_jobs)][i]=F_cj

        # else:
        # # No scheduled jobs: assume "dummy" job completed at time 0
        #     F_Ci = np.zeros(grid_size)
        #     #F_Ci[0] = 1.0

        # --- Unscheduled jobs ---
        unscheduled_jobs = [(i,0) for i in range(self.num_jobs) if (i,0) not in scheduled_jobs]

        for idx in unscheduled_jobs:
            
            F_Ci_temp= np.ones(grid_size)
            
            for i in range(n_machines):
                
                op=(idx[0],i)
                job = next((j for j in self.jobs if j['op'] == op), None)
                # Processing time
                pt_rv = discrete_triangular_rv_given_param(job["a"],job["b"], job["c"])
                f_j = pmf_from_rv(pt_rv, grid_size)
                    
                if i==0:
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
                    
                else:
                    
                    F_rj = F_Ci_temp

                # Start time distribution = max(C_i, r_j) -> lower bound as product
                if scheduled_jobs:
                    F_sj = self.cdfs_estimate[tuple(scheduled_jobs)][i] * F_rj
                else:
                    F_sj = F_rj

                # Completion time distribution
                F_cj = convolve_cdf_with_pmf(F_sj, f_j)

                if i==n_machines-1:
                    # Lateness distribution
                    d_j = job["dd"]
                    F_lj = np.ones(grid_size)
                    if d_j < grid_size:
                        F_lj[:grid_size - d_j] = F_cj[d_j:]

                    F_Lj_list.append(F_lj)

                # Update F_Ci
                F_Ci_temp = F_cj

        # --- Final step: maximum lateness CDF is the product of individual lateness CDFs ---
        F_Lmax_lower_bound = np.ones(grid_size)
        for F_lj in F_Lj_list:
            F_Lmax_lower_bound *= F_lj
            
        return grid, F_Lmax_lower_bound

    def compute_upper_bound_max_lateness_cdf(self, scheduled_jobs, n_machines=1, grid_size=2500):
            
        #TODO: generalizzare a più macchine -> se la macchina è unica, fa semplicemente quanto fatto in precedenza
        #altrimenti, considera come scheduled_jobs quelli schedulati fino all'ultima operazione, mentre stima i tempi 
        #di fine degli altri
        
        """
        Compute the upper bound of the maximum lateness CDF given a partial schedule.

        Returns:
            grid (np.ndarray): Time points.
            F_Lmax_lower_bound (np.ndarray): Lower bound CDF of maximum lateness.
        """
        grid = np.arange(grid_size)
        F_Lj_list = []

        # --- Scheduled jobs first ---
        if scheduled_jobs:
            
            if len(scheduled_jobs)>1:
                
                for j in range(1,len(scheduled_jobs)):
                    
                    idx=scheduled_jobs[j-1]
                    job = next((j for j in self.jobs if j['op'] == idx), None)

                    d_j = job["dd"]
                    F_lj = np.ones(grid_size)
                    
                    n=len(scheduled_jobs)
                    
                    if d_j < grid_size:
                        F_lj[:grid_size - d_j] = self.cdfs_estimate[tuple(scheduled_jobs[:j])][n_machines-1][d_j:]

                    F_Lj_list.append(F_lj)
                    
            idx=scheduled_jobs[-1]

            for i in range(n_machines):
                
                op=(idx[0],i)
                job = next((j for j in self.jobs if j['op'] == op), None)
                # Processing time
                pt_rv = discrete_triangular_rv_given_param(job["a"],job["b"], job["c"])
                f_j = pmf_from_rv(pt_rv, grid_size)
                    
                if i==0:
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
                    
                else:
                    
                    F_rj = self.cdfs_estimate[tuple(scheduled_jobs)][i-1]

                # Start time distribution = max(C_i, r_j) -> lower bound as product
                if len(scheduled_jobs)>1:       
                    F_sj = self.cdfs_estimate[tuple(scheduled_jobs[:-1])][i] * F_rj
                else:
                    F_sj = F_rj

                # Completion time distribution
                F_cj = convolve_cdf_with_pmf(F_sj, f_j)

                if i==n_machines-1:
                    # Lateness distribution
                    d_j = job["dd"]
                    F_lj = np.ones(grid_size)
                    if d_j < grid_size:
                        F_lj[:grid_size - d_j] = F_cj[d_j:]

                    F_Lj_list.append(F_lj)

                # # Update F_Ci
                # F_Ci = F_cj
                
                
                if i==0:
                    schedule_key = tuple(scheduled_jobs)
                    self.cdfs_estimate[schedule_key] = [None] * n_machines
                    
                
                self.cdfs_estimate[tuple(scheduled_jobs)][i]=F_cj

        # else:
        # # No scheduled jobs: assume "dummy" job completed at time 0
        #     F_Ci = np.zeros(grid_size)
        #     #F_Ci[0] = 1.0

        # --- Unscheduled jobs ---
        unscheduled_jobs = [(i,0) for i in range(self.num_jobs) if (i,0) not in scheduled_jobs]
        
        job = next((j for j in self.jobs if j['op'] == unscheduled_jobs[0]), None)
        
        if job['op'][1]==0:
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
                job = next((j for j in self.jobs if j['op'] == idx), None)

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
            
            F_Ci_temp=np.ones(grid_size)
            
            for j in range(n_machines):
                
                op=(idx[0],j)
                job = next((el for el in self.jobs if el['op'] == op), None)
                
                F_C_AminusSminusj = np.ones(grid_size)
                k=0
                
                for i in unscheduled_jobs:
                    
                    if i != idx: 
                        
                        op_i=(i[0],j)
                        job_i = next((el for el in self.jobs if el['op'] == op_i), None)
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
                
                if j==0:
                
                    # Start time lower bound = product
                    F_C_Aminusj_UB =  convolve_cdf_with_pmf((F_Ci_temp*F_rmax),f_C_AminusSminusj)
                
                    pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
                    f_j = pmf_from_rv(pt_rv, grid_size)
                    F_j = compute_cdf(f_j)

                    # Completion time lower bound
                    F_cj_UB = convolve_cdf_with_pmf(F_C_Aminusj_UB, f_j)
                
                else: 
                    
                    # Start time lower bound = product
                    F_C_Aminusj_UB =  convolve_cdf_with_pmf(F_Ci_temp,f_C_AminusSminusj)
                
                    pt_rv = discrete_triangular_rv_given_param(job["a"], job["b"], job["c"])
                    f_j = pmf_from_rv(pt_rv, grid_size)
                    F_j = compute_cdf(f_j)

                    # Completion time lower bound
                    F_cj_UB = convolve_cdf_with_pmf(F_C_Aminusj_UB, f_j)
                
                F_Ci_temp = F_cj_UB

                if j==n_machines-1:
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