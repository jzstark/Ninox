# Runs the whole simulation

# Requirement : I need to add more tables, without running the basic simulation again. The thing is that sampling size and network size might need to be adjusted.

# All the considered barriers: ASP, BSP, pBSP-1/2/3/4.., SSP-2/3/5/10, pSSP-[2/3/5/10]-[1/2/3/4/...]

import barrier
import database as db

"""
Fixed test value
"""

base_exec_time = 5
base_trans_time = 1
stop_time = 120
exec_randomness=0.01
trans_randomness=0.01


# Utils

def randomized_base_speed(speed, straggler_perc, straggleness):
    return speed

def randomized_speed(base_speed, randomness):
    return base_speed

# Data strucutres: node and network

class Node:
    def __init__(self, exec_time):
        self.iteration = 0
        self.clock = 0.

    def calc_time():
        t = randomized_speed(exec_time, exec_randomness)
        self.clock += t
        return t

    def trans_time():
        t = randomized_speed(trans_time, trans_randomness)
        self.clock += t
        return t

    def increase_iter():
        self.iteration += 1


    def get_iteration(self):
        return self.iteration


class Network:

    def __init__(self, config):
        nodes = []
        for i in range(config[size]):
            base_speed = randomized_base_speed(
                base_exec_time,
                config[straggler_perc],
                config[straggleness]
                )
            node = Node(base_speed)
            nodes.append(node)
        self.nodes = nodes


    def execute(self):
        # This is a relaxed stop condition
        while(node.clock < self.stop_time):
            for node in self.nodes:
                t_calc  = node.calc_time()
                t_trans = node.trans_time()
                node.increase_iter()

                db.write(t_run, db, table1)
                db.write(t_wait, db, table2)

"""
A series of postprocessings to add new tables in the db
"""

def pp_wait_time(dbconfig):
    db.connect(dbconfig)
    tb_runtime, tb_trans_time = ...

    for each node:
        tr, tt = tb_runtime[i], tb_trans_time[i]
        for b in barriers:
            wait_time = h(tr, tt, b, some global knowledge)
            write to each barrier tables


def pp_whole_time(): pass

def pp_count_to_time(): pass

def pp_sequence(): pass


def postprocess_db(dbconfig):
    pp_wait_time(dbconfig)


# Entry point
def run(config):
    network = Network(config)
    network.execute()
    postprocess_db(db.dbconfig)
