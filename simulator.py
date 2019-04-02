import numpy as np
import math
import database as db
import utils
import csv
import os
import random

stop_time = 100
randomness=0.01
seed = 666

# Utils

def randomized_speed(base_speed, randomness):
    return base_speed

def random_task_time(straggler_perc, straggleness):
    t = np.random.exponential(1)
    if np.random.uniform() < (straggler_perc / 100.):
        t = t * straggleness
    return t

# Barriers, deciding if `node` should just go ahead

def asp(net, node):
    return True


def bsp(net, node):
    def f(m):
        return (m.step > node.step) or \
            (m.step == node.step and net.clock >= m.t_exec)
    # Do NOT use all() here
    for m in net.nodes:
        if not f(m): return False
    return True


def ssp(staleness):
    def ssp_param(net, node):
        def f(m):
            return (m.step > node.step) or \
                (node.step - m.step < staleness) or \
                (node.step - m.step == staleness and net.clock >= m.t_exec)
        for m in net.nodes:
            if not f(m): return False
        return True
    return ssp_param


def pbsp(sample_size):
    def pbsp_param(net, node):
        # Do NOOOOT use `numpy.random.choice`!!!!
        # It permutes the array each time we call it.
        sampled_nodes = random.sample(net.nodes, sample_size)
        def f(m):
            return (m.step > node.step) or \
                (m.step == node.step and net.clock >= m.t_exec)
        for m in sampled_nodes:
            if not f(m): return False
        return True
    return pbsp_param


def pssp(staleness, sample_size):
    def pssp_param(net, node):
        sampled_nodes = random.sample(net.nodes, sample_size)
        def f(m):
            return (m.step > node.step) or \
                (node.step - m.step < staleness) or \
                (node.step - m.step == staleness and net.clock >= m.t_exec)
        for m in net.nodes:
            if not f(m): return False
        return True
    return pssp_param


# Data strucutres: node and network

class Node:
    def __init__(self):
        self.step   = 0
        self.t_wait = 0.
        self.t_exec = 0.

        self.frontier = [] # length: nodes number
        self.frontier_info = [] # length: total step number


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
        self.step_frontier = [0] * config['size']

        self.straggler_perc = config['straggler_perc']
        self.straggleness = config['straggleness']

        # a potentially very large list; millions of elements
        self.sequence = []

    def update_nodes_time(self):
        for i, n in enumerate(self.nodes):
            if self.clock < n.t_exec or not self.barrier(self, n):
                continue
            # If it's time to finish the wait and go on...

            # Decide my next execution time; increase my step
            exec_time = random_task_time(
                self.straggler_perc, self.straggleness)
            n.t_wait = self.clock
            n.t_exec = n.t_wait + exec_time
            n.step += 1

            # The noisy update from my point of view.
            diff_num = 0 # total deviation from previous step
            diff_max = 0 # max deviation
            for j, s in enumerate(n.frontier):
                diff = self.nodes[j].step - s
                diff_num += diff
                diff_max = diff if diff > diff_max else diff_max

            # Update my progress to ps
            self.step_frontier[i] = n.step
            # Get current screenshot of current progress of all nodes
            n.frontier = self.step_frontier

            if ('frontier' in self.observe_points):
                n.frontier_info.append((diff_num, diff_max))

            if ('sequence' in self.observe_points):
                self.sequence.append((i, n.step, n.t_exec))


    def next_event_at(self):
        t = math.inf
        for n in self.nodes:
            if (n.t_exec > self.clock and n.t_exec < t):
                t = n.t_exec
        return t


    def execute(self):
        np.random.seed(seed)
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

        if ('frontier' in self.observe_points):
            print('Processing frontier: ' + self.dbfilename_sequence)
            self.collect_frontier_data()


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
            self.sequence.sort(key=(lambda x : x[-1]))
            idx, steps, ts = zip(*(self.sequence))
            writer.writerow(idx)
            writer.writerow(steps)
            writer.writerow(ts)


    def collect_frontier_data(self):
        node = self.nodes[0]
        diff_sum, diff_max = zip(*(node.frontier_info))
        print(diff_sum, diff_max)


# Entry point
def run(config):
    for b in config['barriers']:
        network = Network(config, b)
        network.execute()
