from utils import *
import time
import matplotlib.pyplot as plt
from itertools import permutations
from networkx.drawing.nx_pydot import graphviz_layout
import re
import networkx as nx
import os
import matplotlib.image as mpimg
from collections import defaultdict

def build_schedule_with_bounds(instance, n_machine=1, threshold=0.975, plot=False, save_plots_folder=None):

    #numero di job nel job shop (passano tutti per tutte le macchina)
    num_jobs = instance["num_jobs"]
    
    #numero macchine totali
    n_machines = len(instance) - 1
    
    #operazioni da considerare
    if n_machine==1:
        jobs = instance["jobs1"]
    else: jobs = instance[f"jobs{n_machine}"]
    
    #TODO: per ogni macchina, devo andare a calcolare i best schedule finali e rimuovere quelli che hanno lb >= ub best al momento 
    #TODO : poi, vado a estendere il nodo migliore (con schedule completo) sulla macchina successiva e procedo come prima
    #TODO : salvare i best schedule per ogni macchina! 
    #TODO : capire come collegare fine prima macchina e inizio seconda
    
    #nodi "aperti" per ciascun livello, indicati come (livello, schedule, job rimanenti)
    surviving_children_map = {0: [(0, [], list(range(num_jobs)))]}
    
    #stessa mappa, ma senza i best nodes (altrimenti nasce un loop)
    #surviving_children_map = {}
    best_schedule = None
    best_ub = float('inf')

    visited_nodes_count = 0
    visited_levels = set()
    best_partial_schedules = {}
    current_level = 0
    ub_updated_level = 0
    start_time = time.time()

    if save_plots_folder and not os.path.exists(save_plots_folder):
        os.makedirs(save_plots_folder)

    G = nx.DiGraph()
    node_id = 0
    node_map = {}  # maps (tuple_sched, level) to node id

    while current_level in surviving_children_map and surviving_children_map[current_level]:
        nodes_at_level = list(surviving_children_map[current_level])
        
        if current_level!=0 and best_node_data is not None: 
            surviving_children_map[current_level].remove((best_node_data[2], best_node_data[0], best_node_data[3]))

        
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
        best_node_data = None

        for lb, sched, rem in considered_nodes:
            parent_key = (tuple(sched), current_level)
            if parent_key not in node_map:
                node_map[parent_key] = node_id
                G.add_node(node_id, label=str(sched))
                parent_id = node_id
                node_id += 1
            else:
                parent_id = node_map[parent_key]

            if len(rem) <= 2:
                best_local_lateness = float('inf')
                best_local_schedule = None

                for perm in permutations(rem):
                    visited_nodes_count += 1
                    final_schedule = sched + list(perm)
                    grid, F = compute_max_lateness_cdf(jobs, final_schedule, grid_size=2000)
                    idx = np.where(F >= threshold)[0]
                    lateness = grid[idx[0]] if len(idx) > 0 else float('inf')

                    if lateness < best_local_lateness:
                        best_local_lateness = lateness
                        best_local_schedule = final_schedule

                    if plot:
                        plt.figure()
                        plt.plot(grid, F,color='blue', label="CDF")
                        plt.axhline(threshold, color='grey', linestyle='--')
                        plt.text(0.02, threshold + 0.002, f"alpha = {1 - threshold:.3f}", color='red', fontsize=9, verticalalignment='bottom')   
                        if len(idx) > 0:
                            plt.axvline(lateness, color='blue', linestyle='--', label=f"UB = {lateness:.2f}")
                        plt.title(f"Schedule {final_schedule} (leaf) with LB = UB = {lateness:.2f}")
                        plt.xlabel("Max Lateness")
                        plt.ylabel("CDF")
                        plt.legend()
                        if save_plots_folder:
                            plt.savefig(os.path.join(save_plots_folder, f"node_{visited_nodes_count}_leaf.png"), dpi=200)
                            plt.close()
                            
                    child_key = (tuple(final_schedule), current_level + 1)
                    if child_key not in node_map:
                        node_map[child_key] = node_id
                        G.add_node(node_id, label=str(final_schedule))
                        G.add_edge(parent_id, node_id,lb=lb_val, ub=ub_val)
                        node_id += 1

                best_ub = best_local_lateness
                best_schedule = best_local_schedule
                ub_updated_level = current_level
                #best_node_data = (best_schedule, best_local_lateness, best_local_lateness, [])
                #current_level+=1
                print(f"New global best UB {best_ub:.3f} with complete schedule {best_schedule}")
                continue

            for job in rem:
                visited_nodes_count += 1
                new_sched = sched + [job]
                new_rem = [j for j in rem if j != job]

                grid_lb, F_lb = compute_lower_bound_max_lateness_cdf(jobs, new_sched, grid_size=2000)
                grid_ub, F_ub = compute_upper_bound_max_lateness_cdf(jobs, new_sched, grid_size=2000)

                idx_lb = np.where(F_lb >= threshold)[0]
                lb_val = grid_lb[idx_lb[0]] if len(idx_lb) > 0 else float('inf')

                idx_ub = np.where(F_ub >= threshold)[0]
                ub_val = grid_ub[idx_ub[0]] if len(idx_ub) > 0 else float('inf')

                if plot:
                    plt.figure()
                    plt.plot(grid_lb, F_lb, label="Lower Bound CDF",color='red', linestyle='-')
                    plt.plot(grid_ub, F_ub, label="Upper Bound CDF",color='black', linestyle='-')
                    plt.axhline(threshold, color='grey', linestyle='--')
                    plt.text(0.02, threshold + 0.002, f"alpha = {1 - threshold:.3f}", color='grey', fontsize=9, verticalalignment='bottom')
                    if len(idx_lb) > 0:
                        plt.axvline(lb_val, color='red', linestyle='--')
                    if len(idx_ub) > 0:
                        plt.axvline(ub_val, color='black', linestyle='--')
                    plt.fill_between(grid_ub, F_lb, F_ub, where=(F_lb > F_ub), interpolate=True, color='lightgrey', alpha=0.4, label='Bound Range')
                    plt.title(f"Schedule {new_sched} with LB = {lb_val:.2f} and UB = {ub_val:.2f}")
                    plt.xlabel("Max Lateness")
                    plt.ylabel("CDF")
                    plt.legend()
                    if save_plots_folder:
                        plt.savefig(os.path.join(save_plots_folder, f"node_{visited_nodes_count}.png"), dpi=200)
                        plt.close()
                   

                if (best_node_data is None) or (ub_val < best_node_data[1]):
                    best_node_data = (new_sched, ub_val, lb_val, new_rem)

                children.append((lb_val, new_sched, new_rem))

                child_key = (tuple(new_sched), current_level + 1)
                if child_key not in node_map:
                    node_map[child_key] = node_id
                    G.add_node(node_id, label=str(new_sched))
                    G.add_edge(parent_id, node_id,lb=lb_val, ub=ub_val)
                    node_id += 1

        current_main_branch_sched = tuple(best_node_data[0]) if best_node_data else None

        if best_node_data:
            best_ub = best_node_data[1]
            ub_updated_level = current_level
            best_partial_schedules[current_level] = best_node_data[0]
            print(f"New global best UB {best_ub:.3f} with partial schedule {best_node_data[0]}")

        surviving_children = [child for child in children if child[0] <= best_ub]
        pruned_count = len(children) - len(surviving_children)
        if pruned_count > 0:
            print(f"Pruned {pruned_count} children at level {current_level + 1} due to LB >= best_ub")

        #pending_nodes_by_level[current_level + 1] = surviving_children.copy()
        if len(surviving_children)>0:
            surviving_children_map[current_level + 1] = surviving_children.copy()

        if best_node_data and (best_node_data[2], best_node_data[0], best_node_data[3]) in surviving_children_map[current_level + 1]:
            surviving_children_map[current_level + 1].remove((best_node_data[2], best_node_data[0], best_node_data[3]))

        current_level += 1

        for lvl in range(ub_updated_level, current_level):
            if lvl not in surviving_children_map or len(surviving_children_map[lvl]) == 0 or lvl==0:
                continue

            for node in surviving_children_map[lvl]:
                lb, sched, rem = node

                if tuple(sched) == current_main_branch_sched:
                    continue

                if lb >= best_ub:
                    print(f"Back-pruned node at level {lvl} with LB={lb:.3f} >= best_ub={best_ub:.3f}")
                    continue

                temp_sched = sched
                temp_rem = rem
                temp_level = len(temp_sched)

                while temp_level < current_level and temp_rem:
                    
                    best_local = None
                    
                    parent_key = (tuple(temp_sched), temp_level)

                    if parent_key not in node_map:
                        node_map[parent_key] = node_id
                        G.add_node(node_id, label=str(temp_sched[:-1]))
                        parent_id = node_id
                        node_id += 1
                    else:
                        parent_id = node_map[parent_key]
                        
                    if len(temp_rem) <= 2: 
                        
                        best_local_lateness = float('inf')
                        best_local_schedule = None

                        for perm in permutations(temp_rem):
                            
                            visited_nodes_count += 1
                            final_schedule = sched + list(perm)
                            grid, F = compute_max_lateness_cdf(jobs, final_schedule, grid_size=2000)
                            idx = np.where(F >= threshold)[0]
                            lateness = grid[idx[0]] if len(idx) > 0 else float('inf')

                            if lateness < best_local_lateness:
                                best_local_lateness = lateness
                                best_local_schedule = final_schedule

                            if plot:
                                plt.figure()
                                plt.plot(grid, F,color='blue', label="CDF")
                                plt.axhline(threshold, color='grey', linestyle='--')
                                plt.text(0.02, threshold + 0.002, f"alpha = {1 - threshold:.3f}", color='red', fontsize=9, verticalalignment='bottom')   
                                if len(idx) > 0:
                                    plt.axvline(lateness, color='blue', linestyle='--', label=f"UB = {lateness:.2f}")
                                plt.title(f"Schedule {final_schedule} (leaf) with LB = UB = {lateness:.2f}")
                                plt.xlabel("Max Lateness")
                                plt.ylabel("CDF")
                                plt.legend()
                                if save_plots_folder:
                                    plt.savefig(os.path.join(save_plots_folder, f"node_{visited_nodes_count}_leaf.png"), dpi=200)
                                    plt.close()
                                    
                            child_key = (tuple(final_schedule), current_level + 1)
                            if child_key not in node_map:
                                node_map[child_key] = node_id
                                G.add_node(node_id, label=str(final_schedule))
                                G.add_edge(parent_id, node_id,lb=lb_val, ub=ub_val)
                                node_id += 1
                                
                        if best_local_lateness < best_ub: 
                            
                            best_ub = best_local_lateness
                            best_schedule = best_local_schedule
                            temp_rem=[]
                            print(f"Back-tracked new global best UB {best_ub:.3f} with complete schedule {best_schedule}")
                            continue
                    
                    if len(temp_rem)<=2: 
                        temp_level+=1
                        continue
                    
                    for job in temp_rem:
                        visited_nodes_count += 1
                        new_sched = temp_sched + [job]
                        new_rem = [j for j in temp_rem if j != job]

                        grid_ub, F_ub = compute_upper_bound_max_lateness_cdf(jobs, new_sched, grid_size=2000)
                        idx_ub = np.where(F_ub >= threshold)[0]
                        ub_val = grid_ub[idx_ub[0]] if len(idx_ub) > 0 else float('inf')

                        grid_lb, F_lb = compute_lower_bound_max_lateness_cdf(jobs, new_sched, grid_size=2000)
                        idx_lb = np.where(F_lb >= threshold)[0]
                        lb_val = grid_lb[idx_lb[0]] if len(idx_lb) > 0 else float('inf')

                        if best_local is None or ub_val < best_local[1]:
                            best_local = (new_sched, ub_val, lb_val, new_rem)
                            
                        child_key = (tuple(temp_sched), temp_level)
                        
                        if child_key not in node_map:
                            node_map[child_key] = node_id
                            G.add_node(node_id, label=str(temp_sched))
                            G.add_edge(parent_id, node_id,lb=lb_val, ub=ub_val)
                            node_id += 1
                            
                        if plot:
                            plt.figure()
                            plt.plot(grid_lb, F_lb, label="Lower Bound CDF",color='red', linestyle='--')
                            plt.plot(grid_ub, F_ub, label="Upper Bound CDF",color='black', linestyle='-')
                            plt.axhline(threshold, color='grey', linestyle='--')
                            plt.text(0.02, threshold + 0.002, f"alpha = {1 - threshold:.3f}", color='red', fontsize=9, verticalalignment='bottom')
                            if len(idx_lb) > 0:
                                plt.axvline(lb_val, color='red', linestyle='--')
                            if len(idx_ub) > 0:
                                plt.axvline(ub_val, color='black', linestyle='--')
                            plt.fill_between(grid_ub, F_lb, F_ub, where=(F_lb > F_ub), interpolate=True, color='lightgrey', alpha=0.4, label='Bound Range')
                            plt.title(f"Schedule {new_sched} with LB = {lb_val:.2f} and UB = {ub_val:.2f}")
                            plt.xlabel("Max Lateness")
                            plt.ylabel("CDF")
                            plt.legend()
                            if save_plots_folder:
                                plt.savefig(os.path.join(save_plots_folder, f"node_{visited_nodes_count}.png"), dpi=200)
                                plt.close()
                            
                    if best_local:
                        temp_sched, ub_val, lb_val, temp_rem = best_local
                        temp_level += 1

                        if ub_val < best_ub:
                            best_ub = ub_val
                            best_partial_schedules[temp_level - 1] = temp_sched
                            ub_updated_level = temp_level - 1
                            best_node_data = best_local
                            print(f"Updated best UB {best_ub:.3f} from backtracked path with partial schedule {temp_sched}")

                        surviving_children_map[lvl + 1].append((lb_val, temp_sched, temp_rem))
        
        if current_level!=0 and best_node_data is not None: 
            surviving_children_map[current_level].append((best_node_data[2], best_node_data[0], best_node_data[3]))

    elapsed_time = time.time() - start_time
    print("\nBest schedule found:", best_schedule)
    print(f"Visited nodes: {visited_nodes_count}")
    print(f"Best UB (lateness): {best_ub:.3f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    if plot:
        labels = nx.get_node_attributes(G, 'label')
        
        edge_labels = {
            (u, v): f"LB={d['lb']:.2f}\nUB={d['ub']:.2f}"
            for u, v, d in G.edges(data=True)
        }

        # Try to get a top-down layout
        try:
            pos = graphviz_layout(G, prog='dot') 
        except Exception as e:
            print("Graphviz failed, using spring layout:", e)
            pos = nx.spring_layout(G)

        plt.figure(figsize=(20, 14))
        
        # Draw nodes and edges
        nx.draw(G, pos,
                with_labels=False,
                node_color='lightblue',
                edge_color='gray',
                node_size=1200,
                arrows=True)

        # Draw the schedule labels (instead of node IDs)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
        
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
            verticalalignment='center',
        )


        # Formatting
        plt.title("Search Tree", fontsize=16)
        plt.axis('off')
        plt.gca().set_facecolor('lightgrey')

        # Save if requested
        if save_plots_folder:
            tree_plot_path = os.path.join(save_plots_folder, "search_tree.png")
            plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
            print(f"Search tree plot saved to {tree_plot_path}")
            plt.close()
            
            # --- Grouped CDF plots by schedule length (level) ---
    if save_plots_folder:

        # Reverse map node_id to schedule (from node_map)
        node_id_to_sched = {v: k[0] for k, v in node_map.items()}  # node_map keys are (tuple_sched, level)

        grouped_images = defaultdict(list)

        for filename in os.listdir(save_plots_folder):
            if not filename.endswith(".png"):
                continue
            match = re.match(r"node_(\d+).*\.png", filename)
            if not match:
                continue
            node_id = int(match.group(1))
            if node_id not in node_id_to_sched:
                continue
            sched = node_id_to_sched[node_id]
            level = len(sched)
            full_path = os.path.join(save_plots_folder, filename)
            grouped_images[level].append(full_path)

        if grouped_images:
            grouped_images = dict(sorted(grouped_images.items()))  # sort by level

            max_cols = max(len(imgs) for imgs in grouped_images.values())
            num_levels = len(grouped_images)

            fig, axes = plt.subplots(num_levels, max_cols, figsize=(3 * max_cols, 2.5 * num_levels))

            if num_levels == 1:
                axes = [axes]  # ensure iterable

            for row_idx, (level, img_paths) in enumerate(grouped_images.items()):
                for col_idx in range(max_cols):
                    if max_cols > 1:
                        ax = axes[row_idx][col_idx]
                    else:
                        ax = axes[row_idx]

                    if col_idx < len(img_paths):
                        img = mpimg.imread(img_paths[col_idx])
                        ax.imshow(img)
                    ax.axis('off')

            plt.tight_layout()
            combined_path = os.path.join(save_plots_folder, "all_cdfs_by_level.png")
            plt.savefig(combined_path, dpi=300)
            print(f"Saved combined CDF plots by level to {combined_path}")


    return best_schedule, elapsed_time, visited_nodes_count