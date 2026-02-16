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
        self, n_machines=1, plot=False, save_plots_folder=None
    ):
        
        """Builds a schedule for the given instance using a branch and bound algorithm with CDF-based bounds, and generates plots if requested.   
        
        Args:
            n_machine (int, optional): the machine to build the schedule for. Defaults to 1.
            plot (bool, optional): whether to generate plots. Defaults to False.
            save_plots_folder (str, optional): the folder to save the plots in. Defaults to None.
            
        Returns:
            tuple: (best_schedule, elapsed_time) where best_schedule is the best schedule found for the given machine, and elapsed_time is the time taken to find it.
            
        """

        # operations to be considered
        jobs = self.instance["jobs"]

        # initialize the CdF estimator for the given instance 
        self.cdf = CdF_estimator(jobs)

        # TODO: per ogni macchina, devo andare a calcolare i best schedule finali e rimuovere quelli che hanno lb >= ub best al momento
        # TODO : poi, vado a estendere il nodo migliore (con schedule completo) sulla macchina successiva e procedo come prima
        # TODO : salvare i best schedule per ogni macchina!
        # TODO : capire come collegare fine prima macchina e inizio seconda

        # open nodes for each level, given as level: [(lower_bound, partial_schedule, left_out_job),..]
        initial_rem = [(i, j) for i in range(self.num_jobs) for j in range(n_machines)]
        self.surviving_children_map = {0: [(0, [], initial_rem)]}

        # initialize variables to keep track of the best schedule and its upper bound
        best_schedule = None
        best_ub = float("inf")

        # store best partial schedules for each level, to be used for backtracking and pruning of non-main branches. 
        # Given as level: partial_schedule
        best_partial_schedules = {} 
        
        # set current level to 0
        current_level = 0 

        start_time = time.time()

        # results folder
        if save_plots_folder and not os.path.exists(save_plots_folder):
            os.makedirs(save_plots_folder)

        # runs until reaching the leaf nodes
        while current_level < (self.num_jobs - 1):

            # if level is not 0 simply consider the best node found up until now, else consider the starting node
            if current_level != 0 and best_node_data:
                considered_node = [
                    best_node_data[2],
                    best_node_data[0],
                    best_node_data[3],
                ]
            else:
                considered_node = list(self.surviving_children_map[0][0])

            # save children nodes and best node
            children = []
            best_node_data = None

            # starting from the previous level's best node
            lb, sched, rem = considered_node

            # if there are more than 2 jobs remaining
            if len(rem) > 2:

                # try adding each job to the schedule and compute the new bounds
                for job in rem:
                    
                    
                    self.visited_nodes_count += 1
                    new_sched = sched + [job]
                    new_rem = [j for j in rem if j != job]

                    lb_val, ub_val = self.node_evaluation(
                        new_sched, plot, save_plots_folder
                    )


                    # save children if lb_val is smaller than the current best ub_val
                    if best_node_data is None or lb_val < best_node_data[1]:
                        children.append((lb_val, new_sched, new_rem))
                    else:
                        print(
                            f"Pruned child with LB={lb_val:.3f} >= best UB found until now at level {current_level + 1}"
                        )
                        
                    
                    # if ub has improved or one hasn't been found yet for the current level, update it
                    if (best_node_data is None) or (ub_val < best_node_data[1]):
                        best_node_data = (new_sched, ub_val, lb_val, new_rem)
                        
                    

                # if a best node was found, set it as best ub
                if best_node_data:
                    best_ub = best_node_data[1]
                    best_partial_schedules[current_level] = best_node_data[0]
                    print(
                        f"Updated best UB {best_ub:.3f} with partial schedule {best_node_data[0]}"
                    )

                # check if there are children left with lb>=best_ub and prune them
                surviving_children = [
                    child for child in children if child[0] <= best_ub
                ]
                pruned_count = len(children) - len(surviving_children)
                if pruned_count > 0:
                    print(
                        f"Pruned {pruned_count} children at level {current_level + 1} due to LB >= best_ub"
                    )

                # if there are any children left, add them to the map
                if len(surviving_children) > 0:
                    self.surviving_children_map[current_level + 1] = (
                        surviving_children.copy()
                    )

                # remove best_node from the map
                if (
                    best_node_data
                    and (best_node_data[2], best_node_data[0], best_node_data[3])
                    in self.surviving_children_map[current_level + 1]
                ):
                    self.surviving_children_map[current_level + 1].remove(
                        (best_node_data[2], best_node_data[0], best_node_data[3])
                    )

                # increase level
                current_level += 1

            else:  # if there are only two jobs remaining
                
                #find the best complete schedule among the two possible
                best_local_lateness, best_local_schedule = self.final_node_evaluation(
                    sched, rem, plot, save_plots_folder
                )

                # update best ub and level
                best_ub = best_local_lateness
                best_schedule = best_local_schedule
                
                #increase level
                current_level += 1
                print(
                    f"New global best UB {best_ub:.3f} with complete schedule {best_schedule}"
                )

            # backtrack to previous levels
            for lvl in range(1, current_level):

                # if lvl is 0 or there are no children left, proceed to next level
                if (
                    lvl not in self.surviving_children_map
                    or len(self.surviving_children_map[lvl]) == 0
                ):
                    continue

                # for each node, check if it can be pruned and
                for node in self.surviving_children_map[lvl]:
                    lb, sched, rem = node

                    if lb >= best_ub:  # prune the node if lb >= best_ub
                        print(
                            f"Back-pruned node at level {lvl} with LB={lb:.3f} >= best_ub={best_ub:.3f}"
                        )
                        self.surviving_children_map[lvl].remove(node)
                        continue

        for lvl in range(1, current_level):

            best_level = None
            # for each node, check if it can be pruned and, if not, develop it to the current level
            for node in self.surviving_children_map[lvl]:
                lb, sched, rem = node

                if lb >= best_ub:  # prune the node if lb >= best_ub
                    print(
                        f"Back-pruned node at level {lvl} with LB={lb:.3f} >= best_ub={best_ub:.3f}"
                    )
                    self.surviving_children_map[lvl].remove(node)
                    continue

                #initialize temporary variables to keep track of the current node's schedule, remaining jobs and level
                temp_sched = sched
                temp_rem = rem
                
                best_local = None

                if len(temp_rem) > 2:

                    # for each remaining job, add it to the temporary schedule and check results
                    for job in temp_rem:
                        
                        self.visited_nodes_count += 1
                        new_sched = temp_sched + [job]
                        new_rem = [j for j in temp_rem if j != job]

                        lb_val, ub_val = self.node_evaluation(
                            new_sched, plot, save_plots_folder
                            )

                        if best_local is None or ub_val < best_local[1]:
                            best_local = (new_sched, ub_val, lb_val, new_rem)
                            
                        if best_local is None:
                            self.surviving_children_map[lvl + 1].append(
                                (lb_val, new_sched, new_rem)
                            )
                        elif lb_val < best_ub and lb_val < best_local[1]:
                            self.surviving_children_map[lvl + 1].append(
                                (lb_val, new_sched, new_rem)
                            )

                    
                    # if best local was not pruned, add the node to the surviving children map
                    if best_local:
                        temp_sched, ub_val, lb_val, temp_rem = best_local

                        # if found better ub, update all
                        if ub_val < best_ub:
                            best_ub = ub_val
                            best_partial_schedules[lvl] = temp_sched
                            best_node_data = best_local
                            print(
                                f"Updated best UB {best_ub:.3f} from backtracked path with partial schedule {temp_sched}"
                            )

                # if temp_rem only contains two jobs, simply check the two options compared to the best ub found
                else:

                    
                    best_local_lateness, best_local_schedule = (
                        self.final_node_evaluation(
                            temp_sched, temp_rem, plot, save_plots_folder
                        )
                    )

                    #if ub improved, update best ub and schedule
                    if best_local_lateness < best_ub:

                        best_ub = best_local_lateness
                        best_schedule = best_local_schedule
                        temp_rem = []
                        print(
                            f"Back-tracked new global best UB {best_ub:.3f} with complete schedule {best_schedule}"
                        )
                        continue

        elapsed_time = time.time() - start_time
        print("\nBest schedule found:", best_schedule)
        print(f"Visited nodes: {self.visited_nodes_count}")
        print(f"Best UB (lateness): {best_ub:.3f}")
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
