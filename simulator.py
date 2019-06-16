import numpy as np
import math
import database as db
import utils
import csv
import os
import random
import regression_dnn as regression
import gc

randomness = 0.01
seed = 666
modelsize = (28 * 28, 10)
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
        for m in sampled_nodes:
            if not f(m): return False
        return True
    return pssp_param


# Data structures: node and network

class Node:
    def __init__(self, wid, reg):
        self.wid = wid
        self.step   = 0
        self.t_wait = 0.
        self.t_exec = 0.
        # each model has its own copy of model; do not share a common model in computing gradeint.
        if(reg == True):
            opt = regression.make_optimiser()
            self.model = regression.build_model(opt, accuracy=False)
        self.delta = regression.build_update()

        self.frontier = [] # length: nodes number
        self.frontier_info = [] # length: total step number

        self.delta_ready = False # if this node has already calculated an update


class Network:

    def __init__(self, config, barrier):
        self.barrier = barrier[0]
        self.observe_points = config['observe_points']
        # Maintain consistency of datafile names
        self.dbfilename_regression = utils.dbfilename(config,
            barrier[1], 'regression')
        self.dbfilename_step = utils.dbfilename(config, barrier[1], 'step')
        self.dbfilename_sequence = utils.dbfilename(config,
            barrier[1], 'sequence')
        self.dbfilename_frontier = utils.dbfilename(config,
            barrier[1], 'frontier')
        self.dbfilename_ratio = utils.dbfilename(config,
            barrier[1], 'ratio')

        nodes = []
        size = config['size']
        for i in range(size):
            reg_flag = 'regression' in self.observe_points
            node = Node(wid=i, reg=reg_flag)
            nodes.append(node)
        # This could a problem if we allow nodes dropping in and out freely
        self.nodes = nodes
        self.size = size
        self.stop_time = config['stop_time']
        self.clock = 0.

        # Communication delay
        self.delay = [0] * size
        np.random.seed(seed)
        #self.delay[0] = 1
        #self.delay[1] = 5
        for i in range(size):
            #self.delay[i] = np.random.exponential(3)
            self.delay[i] = random.randint(1, 5)
            #self.delay[i] = random.random()


        if('regression' in self.observe_points):
            opt = regression.make_optimiser()
            self.model = regression.build_model(opt)  #np.random.rand(*modelsize)s
        self.regression_info = []

        self.straggler_perc = config['straggler_perc']
        self.straggleness = config['straggleness']
        self.step_frontier = [0] * size
        self.calc_time = [0] * size # total calc time
        # a potentially very large list; millions of elements
        self.sequence = []

        self.rejected_request_1 = 0
        self.rejected_request_2 = 0
        self.rejected_request_3 = 0
        self.accepted_request   = 0


    def print_info(self):
        print("Barrier %s: accepted_request: %d, rejected: %d-%d-%d\n" %
            (self.dbfilename_step, self.accepted_request, self.rejected_request_1, self.rejected_request_2, self.rejected_request_3))


    def update_nodes_time(self):
        passed = []
        for i, n in enumerate(self.nodes):
            if self.clock >= n.t_exec and self.barrier(self, n):
                passed.append((i, n))

        foo = [x for (x, y) in passed]
        print("\npassed: ", foo)

        for i, n in passed:
            self.accepted_request += 1
            # If it's time to finish the wait and go on...

            # Decide my next execution time; increase my step
            # Process time
            exec_time = random_task_time(
                self.straggler_perc, self.straggleness)
            # Communication time
            exec_time *= self.delay[n.wid]
            print("exec_time for worker %i: %.1f." % (n.wid, exec_time))
            n.t_wait = self.clock
            n.t_exec = n.t_wait + exec_time
            n.step += 1

            self.calc_time[i] += exec_time

            # The mismatch everytime before I push the the updates
            if ('frontier' in self.observe_points):
                # The noisy update from my point of view.
                # All the updates that are missing from node n
                if n.frontier == []:
                    diff = self.step_frontier
                else:
                    diff = np.abs(np.subtract(self.step_frontier, n.frontier))
                diff_num = np.sum(diff)
                diff_max = np.max(diff)
                diff_min = np.min(diff)
                # print(diff_num)
                # Node n's one update that is missed by other nodes
                diff_num += 1

                n.frontier_info.append((diff_num, diff_max, diff_min))

            loss, acc = regression.compute_accuracy(self.model)
            print("\nAcc before processing: %.5f" % acc)

            if('regression' in self.observe_points):
                # Push my update to server
                # This step indeed works.
                regression.update_model(self.model, n.delta)
                n.delta_ready = False
            #print("\nNode #", i)
            #print("PS frontier:", self.step_frontier)
            #print("My fronter:", n.frontier)

            loss, acc = regression.compute_accuracy(self.model)
            print("Acc after processing: %.5f" % acc)


        for i, n in passed:
            # Update my progress to ps
            self.step_frontier[i] = n.step

        if('regression' in self.observe_points):
            N = len(self.nodes)
            for i, n in passed:
                weights = regression.get_weight(self.model)
                regression.set_weight(n.model, weights)
                n.delta = regression.compute_updates(n.model, i, N, n.step)

        for i, n in passed:
            # Get current screenshot of current progress of all nodes
            n.frontier = list.copy(self.step_frontier)

            if ('sequence' in self.observe_points):
                self.sequence.append((i, n.step, n.t_exec))


    def update_nodes_delta(self):
        N = len(self.nodes)
        for i, n in enumerate(self.nodes):

            #if self.clock != n.t_wait : continue #!!!!!!

            if n.delta_ready: #or (self.clock > n.t_wait) or (self.clock < n.t_exec) :
                continue

            # Pull weights from parameter server
            weights = regression.get_weight(self.model)
            regression.set_weight(n.model, weights)
            n.delta = regression.compute_updates(n.model, i, N, n.step)

            n.delta_ready = True
            #print("\nFuck: worker %d!\n" % i)
            #print(n.delta)


    def next_event_at(self):
        t = math.inf
        for n in self.nodes:
            if (n.t_exec > self.clock and n.t_exec < t):
                t = n.t_exec
        return t


    def execute(self):
        #np.random.seed(seed)

        counter = 0
        #if ('regression' in self.observe_points):
        #    loss, acc = regression.compute_accuracy(self.model)
        #    self.regression_info.append((0, 0, acc))

        while(self.clock < self.stop_time):
            #if ('regression' in self.observe_points):
            #    self.update_nodes_delta()
            self.update_nodes_time()
            t = self.next_event_at()
            self.clock = t

            if ('regression' in self.observe_points):
                if (self.clock - counter > 1):
                    #max_step = int(np.max(self.step_frontier))
                    avg_step = np.mean(self.step_frontier)
                    loss, acc = regression.compute_accuracy(self.model)
                    self.regression_info.append((self.clock, avg_step, acc))
                    counter = self.clock


        if ('regression' in self.observe_points):
            print('Processing step: ' + self.dbfilename_regression)
            self.collect_regression_data()

        if ('step' in self.observe_points):
            print('Processing step: ' + self.dbfilename_step)
            self.collect_step_data()

        if ('sequence' in self.observe_points):
            print('Processing seq: ' + self.dbfilename_sequence)
            self.collect_sequence_data()

        if ('frontier' in self.observe_points):
            print('Processing frontier: ' + self.dbfilename_frontier)
            self.collect_frontier_data()

        if ('ratio' in self.observe_points):
            print('Processing ratio: ' + self.dbfilename_ratio)
            self.collect_ratio_data()


    def collect_regression_data(self):
        clock, iteration, loss = zip(*self.regression_info)
        filename = self.dbfilename_regression
        with open(filename, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(clock)
            writer.writerow(iteration)
            writer.writerow(loss)


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
        total_diff = []
        total_diff_max = []
        total_diff_min = []
        # Possible memory issue.
        for n in self.nodes:
            diff_sum, diff_max, diff_min = zip(*(n.frontier_info))
            total_diff.extend(diff_sum)
            total_diff_max.extend(diff_max)
            total_diff_min.extend(diff_min)

        filename = self.dbfilename_frontier
        with open(filename, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(total_diff)
            writer.writerow(total_diff_max)
            writer.writerow(total_diff_min)
        # print(np.mean(total_diff), np.std(total_diff))
        #n = self.nodes[0]
        #diff_sum, diff_max = zip(*(n.frontier_info))
        #print(diff_sum)
        #print(np.mean(diff_sum), np.std(diff_sum))

    def collect_ratio_data(self):
        filename = self.dbfilename_ratio
        with open(filename, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(self.calc_time)

# Entry point
def run(config):
    for b in config['barriers']:
        network = Network(config, b)
        network.execute()
        network.print_info()
        del(network)
        #gc.collect()
