from utils import *
from objetive_function_estimators import *
import time
import matplotlib.pyplot as plt
from itertools import permutations
import os


class Branch_Bound():

    def __init__(self, instance, threshold=0.975, obj_f="var"):

        self.threshold = threshold
        self.obj_f = obj_f
        self.visited_nodes_count = 0

        self.instance = instance
        self.num_machines = len(instance)  # numero macchine totali
        self.num_jobs = instance[
            "num_jobs"
        ]  # numero di job nel job shop (passano tutti per tutte le macchina)

    def build_schedule_with_bounds(
        self, n_machines=1, plot=False, save_plots_folder=None, gap_tolerance=0.05
    ):
        """
        Builds a schedule using Depth-First Branch and Bound (DFS).
        Mimics the efficient 'diving' strategy of your old code to find a UB quickly.
        """
        
        # 1. Initialization
        jobs = self.instance["jobs"]
        self.cdf = CdF_estimator(jobs)
        
        start_time = time.time()
        self.visited_nodes_count = 0

        if save_plots_folder and not os.path.exists(save_plots_folder):
            os.makedirs(save_plots_folder)

        # Initialize Best Upper Bound to Infinity
        best_ub = float("inf")
        best_schedule = None

        # Prepare initial state
        initial_rem = [(i, j) for i in range(self.num_jobs) for j in range(n_machines)]
        
        # STACK: LIFO (Last In, First Out) for Depth-First Search
        # Format: (Lower_Bound, Schedule, Remaining_Jobs)
        stack = [(0, [], initial_rem)]
        
        print(f"--- Starting Depth-First Search ---")

        while stack:
            # Pop the latest node (Deepest node) to simulate a "Dive"
            parent_lb, parent_sched, parent_rem = stack.pop()
            
            # PRUNING CHECK
            # 1. Standard: LB >= UB
            if parent_lb >= best_ub:
                continue
            
            # 2. Relative Gap: If improvement is too small, prune
            if best_ub != float("inf"):
                denom = best_ub if best_ub > 0 else 1.0
                if (best_ub - parent_lb) / denom < gap_tolerance:
                    continue

            # --- LEAF OPTIMIZATION (2 Jobs Remaining) ---
            if len(parent_rem) == 2:
                
                if parent_lb >= best_ub:
                    continue
                
                local_lb, local_sched = self.final_node_evaluation(
                    parent_sched, parent_rem, plot, save_plots_folder
                )
                
                if local_lb < best_ub:
                    best_ub = local_lb
                    best_schedule = local_sched
                    print(f"  > New Best Solution (Leaf): {best_ub:.3f} | Schedule: {best_schedule}")
                continue
            
            # --- BRANCHING ---
            children = []
            for job in parent_rem:
                self.visited_nodes_count += 1
                
                new_sched = parent_sched + [job]
                new_rem = [j for j in parent_rem if j != job]

                lb_val, ub_val = self.node_evaluation(
                    new_sched, plot, save_plots_folder
                )

                # Update Global Best UB immediately if found
                if ub_val < best_ub:
                    best_ub = ub_val
                    print(f"  > Updated Global UB to {best_ub:.3f} from partial: {new_sched}")

                # Add to potential children only if promising
                # Check Gap Tolerance for child
                keep_child = True
                if best_ub != float("inf"):
                     if lb_val >= best_ub:
                         keep_child = False
                     else:
                         denom = best_ub if best_ub > 0 else 1.0
                         if (best_ub - lb_val) / denom < gap_tolerance:
                             keep_child = False
                
                if keep_child:
                    children.append((lb_val, new_sched, new_rem))

            # --- CRITICAL STEP: REVERSE SORT ---
            # We want to pop the BEST child first. 
            # Since stack is LIFO, we sort descending (Largest LB first) -> Smallest LB is at the end.
            # This ensures the best node is popped in the very next iteration.
            children.sort(key=lambda x: x[0], reverse=True)
            
            # Push children to stack
            stack.extend(children)

            # Optional: Print progress occasionally
            # if self.visited_nodes_count % 1000 == 0:
            #      print(f"Visited: {self.visited_nodes_count} | Stack Size: {len(stack)} | Best UB: {best_ub:.3f}")

        elapsed_time = time.time() - start_time
        print("\nFinal Best schedule found:", best_schedule)
        print(f"Total Visited nodes: {self.visited_nodes_count}")
        print(f"Final Best UB: {best_ub:.3f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        return best_schedule, elapsed_time
    
    def node_evaluation(self, new_sched, plot=False, folder=None):
        """
        Returns the lower and upper bound for the lateness of a given incomplete schedule, and generates a plot if requested

        Args:
            new_sched (list): incomplete schedule to evaluate
            plot (bool, optional): whether to generate a plot. Defaults to False.
            folder (str, optional): the folder to save the plot in. Defaults to None.

            Returns:
                tuple: (lower_bound, upper_bound) for the given schedule
        """

        # Compute the CDF bounds for the given incomplete schedule
        grid_lb, F_lb = self.cdf.compute_lower_bound_max_lateness_cdf(
            new_sched, grid_size=2500
        )
        grid_ub, F_ub = self.cdf.compute_upper_bound_max_lateness_cdf(
            new_sched, grid_size=2500
        )

        # Find the indices for the quantile
        (idx_lb,) = np.asarray(
            F_lb >= self.threshold
        ).nonzero()  # TODO:modifica lettura
        (idx_ub,) = np.asarray(F_ub >= self.threshold).nonzero()

        # Depending on the objective function, compute the bounds
        if self.obj_f == "c_var":

            # Extract the tail of the distribution starting from the quantile index
            tail_grid_lb = grid_lb[idx_lb]
            tail_F_lb = F_lb[idx_lb]

            # Compute the differences in the CDF values (dF) for the tail, prepending the threshold to account for the jump at the quantile
            du_lb = np.diff(tail_F_lb, prepend=F_lb[idx_lb[0] - 1])

            # Compute CVaR, i.e. the discrete integral (sum of VaR * dF) normalized
            lb_val = round(np.sum(tail_grid_lb * du_lb) / (1 - self.threshold))

            # Do the same for the upper bound
            tail_grid_ub = grid_ub[idx_ub]
            tail_F_ub = F_ub[idx_ub]

            du_ub = np.diff(tail_F_ub, prepend=F_ub[idx_ub[0] - 1])

            ub_val = round(np.sum(tail_grid_ub * du_ub) / (1 - self.threshold))

        elif self.obj_f == "var":
            # For VaR, simply take the grid value at the quantile index, or infinity if the quantile is not reached

            lb_val = grid_lb[idx_lb[0]] if len(idx_lb) > 0 else float("inf")

            ub_val = grid_ub[idx_ub[0]] if len(idx_ub) > 0 else float("inf")

        # generate plot
        if plot:
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.figure()
            plt.plot(grid_lb, F_lb, label="Lower Bound CDF", color="red", linestyle="-")
            plt.plot(
                grid_ub, F_ub, label="Upper Bound CDF", color="black", linestyle="-"
            )
            plt.axhline(self.threshold, color="grey", linestyle="--")
            plt.text(
                0.02,
                self.threshold + 0.002,
                f"alpha = {1 - self.threshold:.3f}",
                color="grey",
                fontsize=9,
                verticalalignment="bottom",
            )
            if len(idx_lb) > 0:
                plt.axvline(lb_val, color="red", linestyle="--")
            if len(idx_ub) > 0:
                plt.axvline(ub_val, color="black", linestyle="--")
            plt.fill_between(
                grid_ub,
                F_lb,
                F_ub,
                where=(F_lb > F_ub),
                interpolate=True,
                color="lightgrey",
                alpha=0.4,
                label="Bound Range",
            )
            plt.title(
                f"Schedule {new_sched} with LB = {lb_val:.2f} and UB = {ub_val:.2f}"
            )
            plt.xlabel("Max Lateness")
            plt.ylabel("CDF")
            plt.legend()
            if folder:
                plt.savefig(
                    os.path.join(folder, f"node_{self.visited_nodes_count}.png"),
                    dpi=200,
                )
                plt.close()

        return lb_val, ub_val

    def final_node_evaluation(self, schedule, rem, plot=False, folder=None):
        """Given a schedule and a list of remaining jobs, evaluates the maximum lateness of all possible complete schedules obtained by permuting the remaining jobs, and returns the best one. Also generates a plot if requested.

        Args:
            schedule (list): the current partial schedule
            rem (list): the list of remaining jobs 
            plot (bool, optional): whether to generate a plot. Defaults to False.        
            folder (str, optional): the folder to save the plot in. Defaults to None.           
            
        Returns:                
            tuple: (best_lateness, best_schedule) 
            
            where best_lateness is the maximum lateness of the best complete schedule found, and best_schedule is the corresponding complete schedule.
        """

        # initialize variables to keep track of the best lateness and schedule
        best_local_lateness = float("inf")
        best_local_schedule = None

        # for each permutation of the remaining jobs, compute the complete schedule and evaluate its maximum lateness using the CDF. If it's better than the best found so far, update the best lateness and schedule. Also generate a plot if requested.
        for perm in permutations(rem):
            
            #increase visited nodes count for each leaf evaluated
            self.visited_nodes_count += 1
            
            #compute complete schedule and evaluate it depending on the objective function
            final_schedule = schedule + list(perm)
            grid, F = self.cdf.compute_max_lateness_cdf(final_schedule, grid_size=2500)
            idx, = np.asarray(F >= self.threshold).nonzero()
            
            if self.obj_f == "c_var":
                
                # Compute the integral of the tail of the distribution 
                
                tail_grid = grid[idx]
                tail_F = F[idx]

                dF = np.diff(tail_F, prepend=self.threshold)

                lateness = round(np.sum(tail_grid * dF) / (1 - self.threshold))

            elif self.obj_f == "var":

                #compute the quantile of the distribution at the threshold
                lateness = grid[idx[0]] if len(idx) > 0 else float("inf")

            # if the lateness of the complete schedule is better than the best found so far, update the best lateness and schedule
            if lateness < best_local_lateness:
                best_local_lateness = lateness
                best_local_schedule = final_schedule

            # print plot
            if plot:
                if folder and not os.path.exists(folder):
                    os.makedirs(folder)
                plt.figure()
                plt.plot(grid, F, color="blue", label="CDF")
                plt.axhline(self.threshold, color="grey", linestyle="--")
                plt.text(
                    0.02,
                    self.threshold + 0.002,
                    f"alpha = {1 - self.threshold:.3f}",
                    color="red",
                    fontsize=9,
                    verticalalignment="bottom",
                )
                if len(idx) > 0:
                    plt.axvline(
                        lateness,
                        color="blue",
                        linestyle="--",
                        label=f"UB = {lateness:.2f}",
                    )
                plt.title(
                    f"Schedule {final_schedule} (leaf) with LB = UB = {lateness:.2f}"
                )
                plt.xlabel("Max Lateness")
                plt.ylabel("CDF")
                plt.legend()
                if folder:
                    plt.savefig(
                        os.path.join(
                            folder, f"node_{self.visited_nodes_count}_leaf.png"
                        ),
                        dpi=200,
                    )
                    plt.close()

        return best_local_lateness, best_local_schedule
