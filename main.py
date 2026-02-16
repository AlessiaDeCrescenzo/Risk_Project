from utils import *
from process_file import *
from solvers.algorithm import Branch_Bound
from Instances import *


#instances=sample_test_instances(num_instances=32, seed=1)
#instances=process_file("test_10_01.txt")

instances=read_file_fs("test_10_01.txt", n_machines=1)

toy_instance ={
        "num_jobs": 4,
        "jobs": [
            {
                "op":(0,0),
                "a": 63,
                "c": 108,
                "b": 118,
                "rda": 414,
                "rdb": 514,
                "dd": 513
            },
            {
                "op":(1,0),
                "a": 105,
                "c": 161,
                "b": 173,
                "rda": 780,
                "rdb": 870,
                "dd": 827
            },
            {
                "op":(2,0),
                "a": 50,
                "c": 90,
                "b": 99,
                "rda": 120,
                "rdb": 224,
                "dd": 201
            },
            {
                "op":(3,0),
                "a": 105,
                "c": 160,
                "b": 171,
                "rda": 818,
                "rdb": 938,
                "dd": 920
            }
        ]
    }

bandb = Branch_Bound(toy_instance,threshold=0.95, obj_f = "var")
best_schedule,t = bandb.build_schedule_with_bounds(plot=True, save_plots_folder='plots_toy_example')

#bandb_c_var= Branch_Bound(toy_instance,threshold=0.95,obj_f= "c_var")
#best_schedule,t = bandb_c_var.build_schedule_with_bounds(plot=True, save_plots_folder='plots_toy_example')

mean_time=0
mean_n=0
mean_time_c_var=0
mean_n_c_var=0
for i in range(32):
    
    var_solver = Branch_Bound(instances[i],threshold=0.95, obj_f = "var")
    #c_var_solver = Branch_Bound(instances[i],threshold=0.95,obj_f= "c_var")
    
    best_schedule,t = var_solver.build_schedule_with_bounds()
    #best_schedule_c_var,t_c_var = c_var_solver.build_schedule_with_bounds()
    mean_time+=t
    mean_n+=var_solver.visited_nodes_count
    #mean_time_c_var+=t_c_var
    #mean_n_c_var+=c_var_solver.visited_nodes_count
    
mean_time=mean_time/32
mean_n=mean_n/32

# mean_time_c_var=mean_time_c_var/32
# mean_n_c_var=mean_n_c_var/32

print(f"Mean Elapsed Time for instance:{mean_time}")
print(f"Mean Number of Visited Nodes for instance:{mean_n}")
