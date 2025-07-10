import numpy as np
from scipy.stats import rv_discrete
from utils import *
from process_file import *
from algorithm import build_schedule_with_bounds
from Instances import *
import matplotlib.pyplot as plt

instances=process_file("test_20_01.txt")

toy_instance ={
        "num_jobs": 4,
        "jobs": [
            {
                "a": 63,
                "c": 108,
                "b": 118,
                "rda": 414,
                "rdb": 514,
                "dd": 513
            },
            {
                "a": 105,
                "c": 161,
                "b": 173,
                "rda": 780,
                "rdb": 870,
                "dd": 827
            },
            {
                "a": 50,
                "c": 90,
                "b": 99,
                "rda": 120,
                "rdb": 224,
                "dd": 201
            },
            {
                "a": 105,
                "c": 160,
                "b": 171,
                "rda": 818,
                "rdb": 938,
                "dd": 920
            }
        ]
    }

best_schedule,t,visited_n = build_schedule_with_bounds(toy_instance, threshold=0.95,plot=True, save_plots_folder='plots_toy_example')

mean_time=0
mean_n=0
for i in range(32):
    best_schedule,t,visited_n = build_schedule_with_bounds(instances[i], threshold=0.95)
    mean_time+=t
    mean_n+=visited_n
    
mean_time=mean_time/32
mean_n=mean_n/32

print(f"Mean Elapsed Time for instance:{mean_time}")
print(f"Mean Number of Visited Nodes for instance:{mean_n}")
