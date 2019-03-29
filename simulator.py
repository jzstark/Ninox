import numpy as np
import math
import database as db
import utils
import csv

stop_time = 50
randomness=0.01

"""
observation point:
- "step" : final step of all nodes
"""

# Utils

def randomized_speed(base_speed, randomness):
    return base_speed

def random_task_time(straggler_perc, straggleness):
    t = np.random.exponential(1)
    if np.random.uniform() < (straggler_perc / 100.):
        t += t * (straggleness / 100.)
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
        self.barrier = barrier[0]
        self.observe_points = config['observe_points']
        self.db_basename = utils.config_to_string(config, barrier[1])

        nodes = []
        for i in range(config['size']):
            node = Node()
            nodes.append(node)
        self.stop_time = stop_time
        self.nodes = nodes
        self.clock = 0.


    def update_nodes_time(self):
        for n in self.nodes:
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

        if ('ob_step' in self.observe_points):
            self.collect_step_data()


    def collect_step_data(self):
        result = []
        for n in self.nodes:
            result.append(n.step)

        filename = self.db_basename + '_step.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


# Entry point
def run(config):
    for b in config['barriers']:
        network = Network(config, b)
        network.execute()


config = {'size':100, 'straggler_perc':0., 'straggleness':0., 'barriers':[(pssp(4, 10), 'pssp_s4_p10')], 'observe_points':['ob_step'], 'path':'/Users/stark/Code/Ninox/data'}
run(config)
