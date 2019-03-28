import barrier
import database as db
import numpy as np

"""
Fixed test value
"""

base_exec_time = 5
base_trans_time = 1
stop_time = 120
exec_randomness=0.01
trans_randomness=0.01


# Utils

def randomized_speed(base_speed, randomness):
    return base_speed

def random_task_time(straggler_perc, straggleness):
    t = np.random.exponential(1)
    if numpy.random.uniform() < straggler_perc:
        t = t * straggleness
    return t

# Data strucutres: node and network

class Node:
    def __init__(self, exec_time):
        self.step   = 0
        self.t_wait = 0.
        self.t_exec = 0.

class Network:

    def __init__(self, config):
        nodes = []
        for i in range(config[size]):
            node = Node()
            nodes.append(node)
        self.nodes = nodes
        self.clock = 0.


    def update_nodes_time(self):
        for n in self.nodes:
            if self.clock < n.t_exec or not self.barrier(x, n): continue
            # do some log
            exec_time = random_task_time(
                config[straggler_perc]，config[straggleness])
            n.t_wait = self.clock
            n.t_exec = n.t_wait + exec_time
            n.step += 1


    def next_event_at(self):
        t = math.inf
        for n in self.nodes:
            if (n.t_exec > self.clock and n.t_exec < t):
                t = n.t_wait
        return t


    def execute(self):
        while(node.clock < self.stop_time):
            update_nodes_time()
            t = next_event_at()
            self.clock = t

"""
A series of postprocessings to add new tables in the db


def pp_wait_time(dbconfig):
    db.connect(dbconfig)
    tb_runtime, tb_trans_time = ...

    for each node:
        tr, tt = tb_runtime[i], tb_trans_time[i]
        for b in barriers:
            wait_time = h(tr, tt, b, some global knowledge)
            write to each barrier tables

def postprocess_db(dbconfig):
    pp_wait_time(dbconfig)
"""

# Entry point
def run(config):
    for b in config[barriers]:
        network = Network(config)
        network.execute()
