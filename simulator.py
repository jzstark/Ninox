import database as db
import numpy as np
import math

stop_time = 100
randomness=0.01

# Utils

def randomized_speed(base_speed, randomness):
    return base_speed

def random_task_time(straggler_perc, straggleness):
    t = np.random.exponential(1)
    if np.random.uniform() < straggler_perc:
        t = t * straggleness
    return t

# Barriers

def asp(net, node):
    return True


def bsp(net, node):
    def f(m):
        return (m.step > node.step) or \
            (m.step == node.step and net.clock >= m.t_exec)
    for m in net.nodes:
        if not f(m): return False
    return True


def ssp(staleness):
    def ssp_param(net, node):
        slowest_step = math.inf
        for m in net.nodes:
            if m.step < slowest_step: slowest_step = m.step
        return (node.step - slowest_step <= staleness)
    return ssp_param


def pbsp(sample_size):
    def pbsp_param(net, node):
        def f(m):
            return (m.step > node.step) or \
                (m.step == node.step and net.clock >= m.t_exec)
        sampled_nodes = np.random.choice(net.nodes,
            size=sample_size, replace=False)
        for m in sampled_nodes:
            if not f(m): return False
        return True
    return pbsp_param


def pssp(staleness, sample_size):
    def pssp_param(net, node):
        def ssp_param(net, node):
            sampled_nodes = np.random.choice(net.nodes,
                size=sample_size, replace=False)
            slowest_step = math.inf
            for m in sampled_nodes:
                if m.step < slowest_step: slowest_step = m.step
            return (node.step - slowest_step <= staleness)
        return ssp_param
    return pssp_param


# Data strucutres: node and network

class Node:
    def __init__(self):
        self.step   = 0
        self.t_wait = 0.
        self.t_exec = 0.


class Network:

    def __init__(self, config, barrier):
        nodes = []
        for i in range(config['size']):
            node = Node()
            nodes.append(node)
        self.stop_time = stop_time
        self.nodes = nodes
        self.barrier = barrier
        self.clock = 0.
        # self.progress_tbl = 


    def update_nodes_time(self):
        for n in self.nodes:
            # Looks like a potential bottleneck ...
            if self.clock < n.t_exec or not self.barrier(self, n):
                continue
            # log some information here
            exec_time = random_task_time(
                config['straggler_perc'], config['straggleness'])
            n.t_wait = self.clock
            n.t_exec = n.t_wait + exec_time
            n.step += 1


    def next_event_at(self):
        t = math.inf
        for n in self.nodes:
            if (n.t_exec > self.clock and n.t_exec < t):
                t = n.t_exec
        return t


    def execute(self):
        while(self.clock < self.stop_time):
            print("Time: %.2f\n" % (self.clock))
            self.update_nodes_time()
            t = self.next_event_at()
            self.clock = t


# Entry point
def run(config):
    for b in config['barriers']:
        network = Network(config, b)
        network.execute()


#config = {'size':100, 'straggler_perc':0., 'straggleness':0., 'barriers':[pssp(4, 10)]}
#run(config)
