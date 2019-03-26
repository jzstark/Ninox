# Runs the whole simulation

# Requirement : I need to add more tables, without running the basic simulation again. The thing is that sampling size and network size might need to be adjusted.

# All the considered barriers: ASP, BSP, pBSP-1/2/3/4.., SSP-2/3/5/10, pSSP-[2/3/5/10]-[1/2/3/4/...]

import barrier
import database as db

class Node:
    def __init__(self, config):
        self.exec_time  = exec_time
        self.randomness_exec = randomness
        self.randomness_trans = randomness
        self.iteration = 0
        self.clock = 0.

    def calc_time():
        t = f(exec_time, randomness1)
        self.clock += t
        return t

    def trans_time():
        t = g(trans_time, randomness2)
        self.clock += t
        return t

    def increase_iter():
        self.iteration += 1

    def get_iteration(self):
        return self.iteration


class Network:

    def __init__(self, config):
        self.stop_time = config.stop_time

        nodes = []
        for i in range(config.size):
            exec_time = f(config.exec_time, config.straggler_perc,
                config.straggleness)
            node = Node(exec_time, config.randomness1,
                config.randomness2)
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


def run(config):
    config_set = set()
    if config not in config_set:
        config_set.add(config)
        network = Network(config)
        network.execute()

    postprocess_db(config.dbconfig)
