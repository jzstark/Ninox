import numpy as np
import math
import database as db
import utils
import csv
import os

stop_time = 50
randomness=0.01

# Utils

def randomized_speed(base_speed, randomness):
    return base_speed

def random_task_time(straggler_perc, straggleness):
    t = np.random.exponential(1)
    if np.random.uniform() < (straggler_perc / 100.):
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
        self.barrier = barrier[0]
        self.observe_points = config['observe_points']
        # Maintain consistency of datafile names
        self.dbfilename_step = utils.dbfilename(config, barrier[1], 'step')
        self.dbfilename_sequence = utils.dbfilename(config,
            barrier[1], 'sequence')

        nodes = []
        for i in range(config['size']):
            node = Node()
            nodes.append(node)
        self.stop_time = stop_time
        self.nodes = nodes
        self.clock = 0.

        self.straggler_perc = config['straggler_perc']
        self.straggleness = config['straggleness']

        # a potentially very large list; millions of elements
        self.sequence = [[],[],[]]


    def update_nodes_time(self):
        for i, n in enumerate(self.nodes):
            if self.clock < n.t_exec or not self.barrier(self, n):
                continue
            # log some information here
            exec_time = random_task_time(
                self.straggler_perc, self.straggleness)
            n.t_wait = self.clock
            n.t_exec = n.t_wait + exec_time
            n.step += 1

            if ('sequence' in self.observe_points):
                self.sequence[0].append(i)
                self.sequence[1].append(n.step)
                self.sequence[2].append(n.t_exec)


    def next_event_at(self):
        t = math.inf
        for n in self.nodes:
            if (n.t_exec > self.clock and n.t_exec < t):
                t = n.t_exec
        return t


    def execute(self):
        while(self.clock < self.stop_time):
            self.update_nodes_time()
            t = self.next_event_at()
            self.clock = t

        if ('step' in self.observe_points):
            print('Processing step: ' + self.dbfilename_step)
            self.collect_step_data()

        if ('sequence' in self.observe_points):
            print('Processing seq: ' + self.dbfilename_sequence)
            self.collect_sequence_data()
            for i in self.sequence: i = []


    def collect_step_data(self):
        result = []
        for n in self.nodes:
            result.append(n.step)
        filename = self.dbfilename_step
        with open(filename, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


    def collect_sequence_data(self):
        filename = self.dbfilename_sequence
        with open(filename, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(self.sequence[0])
            writer.writerow(self.sequence[1])
            writer.writerow(self.sequence[2])


# Entry point
def run(config):
    for b in config['barriers']:
        network = Network(config, b)
        network.execute()
