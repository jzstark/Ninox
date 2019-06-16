from simulator import *
from utils import *
import database as db

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

font = 16 #'x-large'
params = {'legend.fontsize': font-2,
          #'figure.figsize': (9.5, 6),
         'axes.labelsize': font-4,
         'axes.titlesize': font,
         'xtick.labelsize':font,
         'ytick.labelsize':font}
pylab.rcParams.update(params)

markers = ['.', '^', 'o', '*', '+']
linestyles = ['-', '--', '-.', ':', '-']

"""
observation point (for each barriers&size&straggler config):
- "step" : final step of all nodes. Format: one line, rows are all nodes.
- "sequence": two rows, each row has about (#nodes * average_steps) elements.
    It is the order that each update is generated.
    First row: id of node;
    second row: the step of that node when this update is generated;
    third row: the time this update is generated

- "frontier"
- "ratio"
- "regression"

Parameters spec:
- straggler_perc: int, 0 ~ 100
- straggleness: >= 1.  float, with only 1 digit after the point at most.
"""

"""
Experiment 1: Distribution of final iteration progress.
"""

# - Table 1, across barriers, final status of all nodes. PDF/CDF vs process
# - Adjust straggler percentage. Redo evalulation. Process mean/std line vs percentage.
# - Adjust straggler scale. Redo evaluation. Process mean/std line vs percentage.

def exp_step(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    observe_points = ['step']
    configs = [
        {'stop_time':200, 'size':100, 'straggler_perc':0, 'straggleness':1, 'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}
    ]

    for c in configs: run(c)

    data = {}
    barrier_names = [s for (_, s) in barriers]
    for name in barrier_names:
        filename = utils.dbfilename(configs[0], name, 'step')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data[name] = [int(s) for s in next(reader)]

    fig, ax = plt.subplots(figsize=(10, 4))
    for name in barrier_names:
        ax.hist(data[name], 20, label=name, rwidth=20)
    ax.set_ylim([0, 60])
    plt.xlabel("Simulated time")
    plt.legend()
    plt.show()

    # Also draw the CDF graph


def exp_samplesize(result_dir):
    db.init_db(result_dir)

    barriers = [
        (pbsp(0), 'pbsp_p0'), (pbsp(1), 'pbsp_p1'),
        (pbsp(2), 'pbsp_p2'), (pbsp(4), 'pbsp_p4'),
        (pbsp(8), 'pbsp_p8'), (pbsp(16), 'pbsp_p16'),
        (pbsp(32), 'pbsp_p32'), (pbsp(64), 'pbsp_p64')
    ]
    observe_points = ['step']
    configs = [
        {'stop_time':200, 'size':200, 'straggler_perc':0, 'straggleness':1, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir}
    ]

    #for c in configs: run(c)

    data = {}
    barrier_names = [s for (_, s) in barriers]
    for name in barrier_names:
        filename = utils.dbfilename(configs[0], name, 'step')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data[name] = [int(s) for s in next(reader)]
    print(data)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in barrier_names:
        n, bins, patches = ax.hist(data[name], 500, cumulative=True, histtype='step', label=name)
        patches[0].set_xy(patches[0].get_xy()[:-1])
    ax.set_ylim([0, 200])
    plt.ylabel("CDF")
    plt.xlabel("Simulated time")
    plt.legend(loc="lower right")
    plt.show()


def exp_straggle_perc(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    observe_points = ['step']
    configs = [
        {'stop_time':200, 'size':100, 'straggler_perc':0, 'straggleness':3, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':5, 'straggleness':3, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':10, 'straggleness':3, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':15, 'straggleness':3, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':20, 'straggleness':3, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir}
    ]

    #for c in configs: run(c)

    dict_stragglers = {}
    for b in barriers:
        dict_single_straggler = {}
        for c in configs:
            filename = utils.dbfilename(c, b[1], 'step')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                data = [int(s) for s in next(reader)]
                mu = np.mean(data)
                std = np.std(data)
                dict_single_straggler[c['straggler_perc']] = (mu, std)
        dict_stragglers[b[1]] = dict_single_straggler

    print(dict_stragglers)

    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        y, _ = zip(*ys)
        y = np.divide(y, y[0])
        ax.plot(x, y, marker=markers[c], label=k)
        c += 1
    plt.legend()
    plt.xlabel("Straggle node percentage")
    plt.ylabel("Normalised average iteration progress")
    plt.show()


def exp_straggleness(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    observe_points = ['step']
    configs = [
        {'stop_time':200, 'size':100, 'straggler_perc':5, 'straggleness':1, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':5, 'straggleness':2, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':5, 'straggleness':4, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':5, 'straggleness':6, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':5, 'straggleness':8, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':5, 'straggleness':10, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir}
    ]

    #for c in configs: run(c)

    dict_stragglers = {}
    for b in barriers:
        dict_single_straggler = {}
        for c in configs:
            filename = utils.dbfilename(c, b[1], 'step')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                data = [int(s) for s in next(reader)]
                mu = np.mean(data)
                std = np.std(data)
                dict_single_straggler[c['straggleness']] = (mu, std)
        dict_stragglers[b[1]] = dict_single_straggler

    print(dict_stragglers)

    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        y, _ = zip(*ys)
        y = np.divide(y, y[0])
        ax.plot(x, y, marker=markers[c], label=k)
        c += 1
    plt.legend()
    plt.xlabel("Straggleness of the slow nodes")
    plt.ylabel("Normalised average iteration progress")
    plt.show()


def exp_straggleness_consistency(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    ob = ['frontier']
    t = 100
    s = 100
    configs = [
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':1, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':2, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':6, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':8, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':10, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
    ]

    #for c in configs: run(c)

    dict_stragglers = {}
    for b in barriers:
        dict_single_straggler = {}
        for c in configs:
            filename = utils.dbfilename(c, b[1], 'frontier')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                diff_num = [int(s) for s in next(reader)]
                diff_max = [int(s) for s in next(reader)]
                diff_min = [int(s) for s in next(reader)]
                mu = np.mean(diff_num)
                std = np.std(diff_num)
                dict_single_straggler[c['straggleness']] = (mu, std)
        dict_stragglers[b[1]] = dict_single_straggler

    print(dict_stragglers)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) #figsize=(12, 5))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        mu, std = zip(*ys)
        #mu = np.divide(mu, mu[0])
        #std = np.divide(std, std[0])
        ax1.plot(x, mu, marker=markers[c], label=k)
        ax2.plot(x, std, marker=markers[c], label=k)
        c = (c + 1) % (len(markers))

    ax1.set_xlabel("Straggle node percentage")
    ax1.set_ylabel("Mean of step difference")
    ax1.legend()

    ax2.set_xlabel("Straggle node percentage")
    ax2.set_ylabel("Stddev of step difference")
    ax2.legend()
    plt.show()


"""
Experiment 2: "Accuracy"
"""

# - Sequence length vs (accuracy compared to BSP); which node *generates* a new update. I expect pBSP and pSSP are bounded, but not ASP. The definition of "difference" should follow that in math proof.
# (Note that sequence length itself is "number of updates")
# - Change straggler percentage of pBSP, pSSP. Redo Evaluation,
# - Change straggler scale. Redo.
# - Chnage x-axis to real time for all the above

#
def exp_regression(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (ssp(2), 'ssp_s2'),
        (bsp, 'bsp'),

        #(ssp(10), 'ssp_s10'),
        #(pbsp(1), 'pbsp_p10'),
        #(pbsp(40), 'pbsp_p40'),
        #(pssp(4, 5), 'pssp_s4_p5'),
    ]
    observe_points = ['regression']
    config = {'stop_time':50, 'size':24, 'straggler_perc':0, 'straggleness':1,
    #config = {'stop_time':50, 'size':99, 'straggler_perc':15, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    run(config)

    clock = {}; iteration = {}; loss = {}
    barrier_names = [s for (_, s) in config['barriers']]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'regression')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            clock[barrier] = [float(s) for s in next(reader)]
            iteration[barrier] = [float(s) for s in next(reader)]
            loss[barrier] = [float(s) for s in next(reader)]

    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for barrier in barrier_names:
        ax.plot(clock[barrier], loss[barrier], label=barrier)

    fig, ax = plt.subplots(figsize=(8, 4))
    for barrier in barrier_names:
        ax.plot(iteration[barrier], loss[barrier], label=barrier)

    #plt.xlim([0,100])
    #plt.ylim([19.25,20.75])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    plt.legend()
    plt.show()
    """

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) #figsize=(12, 5))

    for barrier in barrier_names:
        ax1.plot(clock[barrier], loss[barrier], label=barrier_to_label(barrier))
        ax2.plot(iteration[barrier], loss[barrier],
            label=barrier_to_label(barrier))

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Accuracy")
    #ax2.set_xlim([0,50])
    #ax2.set_ylim([0.6,0.9])
    ax2.legend()

    plt.show()


def exp_seqdiff(result_dir):
    db.init_db(result_dir)

    barriers = [
        # (bsp, 'bsp'), --> should be all 0 or very close to it at least
        (asp, 'asp'), (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
    ]

    size = 100
    # Observe N different points in the whole updates sequence
    N = int(size/3) # a random step
    observe_points = ['sequence']
    config = {'stop_time':50, 'size':size, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    #run(config)

    nodes = {}; steps = {}; times = {}
    barrier_names = [s for (_, s) in config['barriers']]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'sequence')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            nodes[barrier] = [int(s) for s in next(reader)]
            steps[barrier] = [int(s) for s in next(reader)]
            times[barrier] = [float(s) for s in next(reader)]

    #print(nodes)
    #print(steps)

    # l: length of target vector; n: number of workers
    def generate_true_seq(l, n):
        seq = [None] * l
        for i in range(l):
            c = int(i / n)
            p = i % n
            # +1 to agree with the step in exp (starting from 1)
            seq[i] = (p, c+1)
        return seq

    result = {}
    for barrier in barrier_names:
        print("Barrier:", barrier)
        length = len(nodes[barrier]) # length of this sequence
        assert(length > N)
        true_seq  = generate_true_seq(length, config['size'])
        noisy_seq = list(zip(nodes[barrier], steps[barrier]))
        #print("True seq:", true_seq)
        #print("Noisy seq:", noisy_seq)

        diffs = []
        for idx in range(N, length + 1, N):
            # Compute difference
            true_set = set(true_seq[0:idx])
            noisy_set = set(noisy_seq[0:idx])
            diff_a_b = true_set.difference(noisy_set)
            diff_b_a = noisy_set.difference(true_set)
            diff = len(diff_a_b) + len(diff_b_a)
            diffs.append((diff, idx))
        result[barrier] = diffs
    #print(result)

    fig, ax = plt.subplots(figsize=(8, 5))
    barrier_names = result.keys()
    k = 0
    for barrier in barrier_names:
        [y, t] = list(zip(*result[barrier]))
        #y = np.divide(y, t) # not a good metric
        ax.plot(t, y, label=barrier) #, linestyle=linestyles[k])
        k = (k + 1) % len(linestyles)
    ax.set_xlabel("Sequence length T (P=%d)" % size)
    ax.set_ylabel("Noisy-True sequence difference")
    plt.legend()
    plt.show()


def exp_straggle_seqdiff(result_dir):
    db.init_db(result_dir)
    barriers = [
        (asp, 'asp'), (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5'),
        #(pbsp(10), 'pbsp_p10'),
        #(pssp(4, 10), 'pssp_s4_p10'),
    ]

    def generate_true_seq(l, n):
        seq = [None] * l
        for i in range(l):
            c = int(i / n)
            p = i % n
            # +1 to agree with the step in exp (starting from 1)
            seq[i] = (p, c+1)
        return seq

    observe_points = ['sequence']
    time = 60; size = 100
    configs = [
        {'stop_time':time, 'size':size, 'straggler_perc':0, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':5, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':15, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':20, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':25, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    #for c in configs: run(c)

    barrier_names = [s for (_, s) in barriers]
    dict_stragglers = {}

    for barrier in barrier_names:
        dict_single_straggler = {}
        for c in configs:
            n = c['size']
            filename = utils.dbfilename(c, barrier, 'sequence')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                nodes = [int(s) for s in next(reader)]
                steps = [int(s) for s in next(reader)]
                times = [float(s) for s in next(reader)]
                dict_single_straggler[c['straggler_perc']] = (nodes, steps, n)
        dict_stragglers[barrier] = dict_single_straggler

    def nsdiff(ns):
        nodes, steps, n = ns
        length = len(nodes)
        true_seq  = generate_true_seq(length, n)
        noisy_seq = list(zip(nodes, steps))
        true_set = set(true_seq)
        noisy_set = set(noisy_seq)
        diff_a_b = true_set.difference(noisy_set)
        diff_b_a = noisy_set.difference(true_set)
        diff = len(diff_a_b) + len(diff_b_a)
        return diff / length

    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        sizes  = list(i.keys())   # sizes
        nslist = list(i.values()) # (nodes, steps) list
        diffs = list(map(nsdiff, nslist))
        ax.plot(sizes, diffs, label=k, marker=markers[c], linestyle=linestyles[c])
        c = c + 1

    ax.set_xlabel("Straggle node percentage")
    ax.set_ylabel("Normalised nosiy-true sequence difference")
    plt.legend()
    plt.show()


def exp_straggleness_seqdiff(result_dir):
    db.init_db(result_dir)
    barriers = [
        (asp, 'asp'), (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5'),
        #(pbsp(10), 'pbsp_p10'),
        #(pssp(4, 10), 'pssp_s4_p10'),
    ]

    def generate_true_seq(l, n):
        seq = [None] * l
        for i in range(l):
            c = int(i / n)
            p = i % n
            seq[i] = (p, c+1)
        return seq

    observe_points = ['sequence']
    time = 60; size = 100

    configs = [
        {'stop_time':time, 'size':size, 'straggler_perc':5, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':5, 'straggleness':2,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':5, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':5, 'straggleness':6,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':5, 'straggleness':8,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':5, 'straggleness':10,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    for c in configs: run(c)

    barrier_names = [s for (_, s) in barriers]
    dict_stragglers = {}

    for barrier in barrier_names:
        dict_single_straggler = {}
        for c in configs:
            n = c['size']
            filename = utils.dbfilename(c, barrier, 'sequence')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                nodes = [int(s) for s in next(reader)]
                steps = [int(s) for s in next(reader)]
                times = [float(s) for s in next(reader)]
                dict_single_straggler[c['straggleness']] = (nodes, steps, n)
        dict_stragglers[barrier] = dict_single_straggler

    def nsdiff(ns):
        nodes, steps, n = ns
        length = len(nodes)
        true_seq  = generate_true_seq(length, n)
        noisy_seq = list(zip(nodes, steps))
        true_set = set(true_seq)
        noisy_set = set(noisy_seq)
        diff_a_b = true_set.difference(noisy_set)
        diff_b_a = noisy_set.difference(true_set)
        diff = len(diff_a_b) + len(diff_b_a)
        return diff / length

    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        sizes  = list(i.keys())   # sizes
        nslist = list(i.values()) # (nodes, steps) list
        diffs = list(map(nsdiff, nslist))
        ax.plot(sizes, diffs, label=k, marker=markers[c], linestyle=linestyles[c])
        c += 1

    ax.set_xlabel("Straggleness")
    ax.set_ylabel("Normalised nosiy-true sequence difference")
    plt.legend()
    plt.show()


def exp_scalability_seqdiff(result_dir):
    db.init_db(result_dir)
    barriers = [
        (asp, 'asp'), (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5'),
        #(pbsp(10), 'pbsp_p10'),
        #(pssp(4, 10), 'pssp_s4_p10'),
    ]

    def generate_true_seq(l, n):
        seq = [None] * l
        for i in range(l):
            c = int(i / n)
            p = i % n
            # +1 to agree with the step in exp (starting from 1)
            seq[i] = (p, c+1)
        return seq

    observe_points = ['sequence']
    time = 60; size = 100
    configs = [
        {'stop_time':time, 'size':50, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':200, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':300, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':400, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':500, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    for c in configs: run(c)

    barrier_names = [s for (_, s) in barriers]
    dict_stragglers = {}

    for barrier in barrier_names:
        dict_single_straggler = {}
        for c in configs:
            n = c['size']
            filename = utils.dbfilename(c, barrier, 'sequence')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                nodes = [int(s) for s in next(reader)]
                steps = [int(s) for s in next(reader)]
                times = [float(s) for s in next(reader)]
                dict_single_straggler[c['size']] = (nodes, steps, n)
        dict_stragglers[barrier] = dict_single_straggler

    def nsdiff(ns):
        nodes, steps, n = ns
        length = len(nodes)
        true_seq  = generate_true_seq(length, n)
        noisy_seq = list(zip(nodes, steps))
        true_set = set(true_seq)
        noisy_set = set(noisy_seq)
        diff_a_b = true_set.difference(noisy_set)
        diff_b_a = noisy_set.difference(true_set)
        diff = len(diff_a_b) + len(diff_b_a)
        return diff / length

    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        sizes  = list(i.keys())   # sizes
        nslist = list(i.values()) # (nodes, steps) list
        diffs = list(map(nsdiff, nslist))
        ax.plot(sizes, diffs, label=k, marker=markers[c], linestyle=linestyles[c])
        c += 1

    ax.set_xlabel("Network sizes")
    ax.set_ylabel("Normalised nosiy-true sequence difference")
    plt.legend()
    plt.show()



def exp_straggle_accuracy(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (bsp, 'bsp'),
        (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5')
    ]
    observe_points = ['regression']
    t = 100
    s = 60
    configs = [
        {'stop_time':t, 'size':s, 'straggler_perc':0, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':15, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':20, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        #{'stop_time':200, 'size':100, 'straggler_perc':25, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        #'path':result_dir},
        #{'stop_time':200, 'size':100, 'straggler_perc':30, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        #'path':result_dir},
    ]

    for c in configs: run(c)

    dict_stragglers = {}
    for b in barriers:
        dict_single_straggler = {}
        for c in configs:
            filename = utils.dbfilename(c, b[1], observe_points[0])
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                clock = [int(s) for s in next(reader)]
                iteration = [int(s) for s in next(reader)]
                loss = [float(s) for s in next(reader)]
                accuracy = loss[-1]
                dict_single_straggler[c['straggler_perc']] = accuracy
        dict_stragglers[b[1]] = dict_single_straggler

    print(dict_stragglers)

    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        y = list(i.values())
        y = (np.divide(y, y[0]) - 1) * 100
        ax.plot(x, y, marker=markers[c], label=k)
        c += 1
    plt.legend()
    plt.xlabel("Percentage of slow nodes")
    plt.ylabel("Accuracy decrease percentage")
    plt.show()


def exp_frontier(result_dir):
    import scipy.stats as stats

    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5'),

        #(ssp(1), 'ssp_s1'),
        #(ssp(10), 'ssp_s10'),
        #(ssp(20), 'ssp_s20'),

        #(pbsp(20), 'pbsp_p20'),
        #(pbsp(50), 'pbsp_p50'),
        #(pbsp(80), 'pbsp_p80'),
        #(pbsp(95), 'pbsp_p95'),
        #(pbsp(99), 'pbsp_p99'),


        #(pssp(4, 20), 'pssp_s4_p20'),
        #(pssp(4, 50), 'pssp_s4_p50'),
        #(pssp(4, 90), 'pssp_s4_p90'),
        #(pssp(4, 95), 'pssp_s4_p95'),
        #(pssp(4, 99), 'pssp_s4_p99'),
        #(pssp(4, 100), 'pssp_s4_p100'),
    ]
    observe_points = ['frontier']
    config = {'stop_time':100, 'size':100, 'straggler_perc':0, 'straggleness':1.,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    run(config)

    diff_num = {}; diff_max = {}; diff_min = {}
    barrier_names = [s for (_, s) in config['barriers'] ]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'frontier')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            diff_num[barrier] = [float(s) for s in next(reader)]
            diff_max[barrier] = [int(s) for s in next(reader)]
            diff_min[barrier] = [int(s) for s in next(reader)]

    #print(diff_num)


    fig, ax = plt.subplots(figsize=(10, 5))
    c = 0
    barriers = [(k, v) for (k, v) in diff_num.items() if k != "bsp"]
    for k, v in barriers:
        print(k, np.mean(v))
        v = np.divide(v, config['size'])
        density = stats.gaussian_kde(v)
        x = np.linspace(0, 5, 250)
        #n, x, _ = ax.hist(v, 200, histtype='step', cumulative=False, label=k)
        ax.plot(x, density(x), linestyle=linestyles[c % len(linestyles)], label=barrier_to_label(k))
        c += 1
    #ax.axvline(x=1, linestyle=linestyles[c], label='bsp', color='m')
    ax.legend()
    ax.set_xlim([0, 5])
    #ax.set_ylim([0, 1])
    ax.set_xlabel('Average step inconsistency per node')
    ax.set_ylabel('Density')
    plt.show()

    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, v in diff_max.items():
        ax.hist(v, 200, normed=1, histtype='step', cumulative=True, label=k)
    ax.legend()
    ax.set_xlabel('Max step difference (size = %d)' % config['size'])
    ax.set_ylabel('CDF')
    plt.show()
    """


def exp_straggle_consistency(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    ob = ['frontier']
    t = 100
    s = 100
    configs = [
        {'stop_time':t, 'size':s, 'straggler_perc':0, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':5, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':15, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':20, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':25, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
    ]

    #for c in configs: run(c)

    dict_stragglers = {}
    for b in barriers:
        dict_single_straggler = {}
        for c in configs:
            filename = utils.dbfilename(c, b[1], 'frontier')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                diff_num = [int(s) for s in next(reader)]
                diff_max = [int(s) for s in next(reader)]
                diff_min = [int(s) for s in next(reader)]
                mu = np.mean(diff_num)
                std = np.std(diff_num)
                dict_single_straggler[c['straggler_perc']] = (mu, std)
        dict_stragglers[b[1]] = dict_single_straggler

    print(dict_stragglers)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) #figsize=(12, 5))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        mu, std = zip(*ys)
        mu = np.divide(mu, mu[0])
        std = np.divide(std, std[0])
        ax1.plot(x, mu, marker=markers[c], label=k)
        ax2.plot(x, std, marker=markers[c], label=k)
        c = (c + 1) % (len(markers))

    ax1.set_xlabel("Straggle node percentage")
    ax1.set_ylabel("Normalised step difference (mean)")
    ax1.legend()

    ax2.set_xlabel("Straggle node percentage")
    ax2.set_ylabel("Normalised step difference (stddev)")
    ax2.legend()
    plt.show()


"""
Experiment 3: Comparison of time used on running/waiting/transmission.
"""

# - Bar chart. Take only final status (or whole process if you want, but the point is not very clear). Compare barriers.

def exp_ratio(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    observe_points = ['ratio']
    config = {'stop_time':200, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    #run(config)

    ratio = {}
    barrier_names = [s for (_, s) in config['barriers']]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'ratio')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            ratio[barrier] = [float(s) for s in next(reader)]

    fig, ax = plt.subplots()
    ax.boxplot(list(ratio.values()), labels=barrier_names)
    ax.set_ylabel("Time utilisation percentage")
    plt.show()



"""
Experiment 4: Scalability
"""

# Leave this to be decided at this stage. Before using notes as x-axis, I need to do some observation of the Accuracy vs. time/iteratio performance under different network size and **sampling size**. Then I can decide what is a good way to present the scability of sampling size.


def exp_scalability(result_dir):
    db.init_db(result_dir)

    ssp_name = 'ssp_s4'
    barriers = [
        (ssp(4), ssp_name),
        (pssp(4, 2), 'pssp_s4_p2'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 15), 'pssp_s4_p15'),
        (pssp(4, 20), 'pssp_s4_p20'),
    ]
    observe_points = ['regression']
    # run 300 seconds
    configs = [
        {'stop_time':200, 'size':50, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':150, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':200, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':250, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':300, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    # for c in configs: run(c)

    clocks = []; iterations = []; losses = []
    for c in configs:
        clock = {}; iteration = {}; loss = {}
        barrier_names = [s for (_, s) in c['barriers']]
        for barrier in barrier_names:
            filename = utils.dbfilename(c, barrier, 'regression')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                clock[barrier] = [int(s) for s in next(reader)]
                iteration[barrier] = [int(s) for s in next(reader)]
                loss[barrier] = [float(s) for s in next(reader)]
        clocks.append(clock)
        iterations.append(iteration)
        losses.append(loss)

    fig, ax = plt.subplots(figsize=(8, 4))

    markers = ['.', '^', 'o', '*', '+']
    ls = ['-', '--', '-.', ':', '-']

    barrier_names = [s for (_, s) in barriers if s != ssp_name]
    sizes = [c['size'] for c in configs]

    """
    for k, barrier in enumerate(barrier_names):
        y_mean = []; y_std = []
        for i, c in enumerate(configs):
            l = min(len(losses[i][barrier]), len(losses[i][ssp_name]))
            y = np.divide(losses[i][barrier][0:l], losses[i][ssp_name][0:l])
            #ax.plot(iterations[i][barrier], losses[i][barrier],
                #label=barrier+ '_' + str(c['size']))
            y_mean.append(np.mean(y))
            y_std.append(np.std(y))
        ax.plot(sizes, y_mean, marker=markers[k], linestyle=ls[k], label=barrier, )
    """
    for k, barrier in enumerate(barrier_names):
        label = 'p=' + (barrier.split('p'))[-1]
        y = []
        for i, c in enumerate(configs):
            r = np.divide(losses[i][barrier][-1], losses[i][ssp_name][-1])
            y.append(r)
        ax.plot(sizes, y, marker=markers[k], linestyle=ls[k], label=label)
    ax.set_ylabel("Ratio of PSSP/ SSP in regression model loss value")
    ax.set_xlabel("Number of nodes")
    plt.grid(linestyle='--', linewidth=1)
    plt.legend()
    plt.show()


def exp_scalability_step(result_dir):
    db.init_db(result_dir)

    ssp_name = 'ssp_s4'
    barriers = [
        (ssp(4), ssp_name),
        (pssp(4, 2), 'pssp_s4_p2'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 20), 'pssp_s4_p20'),
        (pssp(4, 30), 'pssp_s4_p30'),
        (pssp(4, 40), 'pssp_s4_p40'),
    ]
    observe_points = ['step']
    configs = [
        #{'size':50, 'straggler_perc':0, 'straggleness':1,
        #'barriers':barriers, 'observe_points':observe_points,
        #'path':result_dir},
        {'stop_time':200, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        #{'size':150, 'straggler_perc':0, 'straggleness':1,
        #'barriers':barriers, 'observe_points':observe_points,
        #'path':result_dir},
        {'stop_time':200, 'size':200, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        #{'size':250, 'straggler_perc':0, 'straggleness':1,
        #'barriers':barriers, 'observe_points':observe_points,
        #'path':result_dir},
        {'stop_time':200, 'size':300, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':400, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':500, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':200, 'size':600, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    # for c in configs: run(c)

    steps = []
    for c in configs:
        step = {}
        barrier_names = [s for (_, s) in c['barriers']]
        for barrier in barrier_names:
            filename = utils.dbfilename(c, barrier, 'step')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                step[barrier] = np.mean([int(s) for s in next(reader)])
        steps.append(step)

    #print(steps)

    fig, ax = plt.subplots()
    barrier_names = [s for (_, s) in barriers if s != ssp_name]
    sizes = [c['size'] for c in configs]
    for k, barrier in enumerate(barrier_names):
        label = barrier_to_label(barrier)
        y = []
        for i, c in enumerate(configs):
            ratio = np.divide(steps[i][barrier], steps[i][ssp_name])
            y.append(ratio)
        ax.plot(sizes, y, marker=markers[k % len(markers)],
            linestyle=linestyles[k % len(markers)], label=label)
    ax.set_ylabel("Ratio of PSSP step / SSP step progress")
    ax.set_xlabel("Worker number")
    plt.grid(linestyle='--', linewidth=1)

    plt.legend()
    plt.show()


def exp_scalability_consistency(result_dir):
    db.init_db(result_dir)

    ssp_name = 'ssp_s4'
    barriers_ssp = [
        (asp, 'asp'),
        (ssp(4), ssp_name),
        (pssp(4, 2), 'pssp_s4_p2'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 20), 'pssp_s4_p20'),
        (pssp(4, 30), 'pssp_s4_p30'),
        (pssp(4, 40), 'pssp_s4_p40'),
    ]
    barriers = [
        (asp, 'asp'),
        (pbsp(2),  'pbsp_p2'),
        (pbsp(5),  'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(20), 'pbsp_p20'),
        (pbsp(30), 'pbsp_p30'),
        (pbsp(40), 'pbsp_p40'),
    ]
    observe_points = ['frontier']
    configs = [
        {'stop_time':100, 'size':50, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':200, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points, 'path':result_dir},
        {'stop_time':100, 'size':250, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':300, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':350, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':400, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':450, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':500, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    #for c in configs: run(c)

    diffs = []
    for c in configs:
        diff = {}
        barrier_names = [s for (_, s) in c['barriers']]
        for barrier in barrier_names:
            filename = utils.dbfilename(c, barrier, 'frontier')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                diff_num = [int(s) for s in next(reader)]
                diff_max = [int(s) for s in next(reader)]
                diff_min = [int(s) for s in next(reader)]
                mu  = np.mean(diff_num)
                std = np.std(diff_num)
                diff[barrier] = (mu, std)
        diffs.append(diff)

    print(diffs)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) #figsize=(12, 5))
    barrier_names = [s for (_, s) in barriers] #if s != ssp_name]
    sizes = [c['size'] for c in configs]

    for k, barrier in enumerate(barrier_names):
        label = barrier_to_label(barrier)
        y = []
        for i, c in enumerate(configs):
            #ratio = np.divide(diffs[i][barrier][0], diffs[i][ssp_name][0])
            ratio = np.divide(diffs[i][barrier][0], c['size'])
            y.append(ratio)
        ax1.plot(sizes, y, marker=markers[k % len(markers)],
            linestyle=linestyles[k % len(markers)], label=label)

    for k, barrier in enumerate(barrier_names):
        label = barrier_to_label(barrier)
        y = []
        for i, c in enumerate(configs):
            #ratio = np.divide(diffs[i][barrier][1], diffs[i][ssp_name][1])
            ratio = np.divide(diffs[i][barrier][1], c['size'])
            y.append(ratio)
        ax2.plot(sizes, y, marker=markers[k % len(markers)],
            linestyle=linestyles[k % len(markers)], label=label)

    #ax1.set_ylabel("Ratio of PSSP diff / SSP step consistency (mean) ")
    ax1.set_ylabel("Normalised pBSP step consistency (mean) ")
    ax1.set_xlabel("Worker number")
    ax1.legend()

    #ax2.set_ylabel("Ratio of PSSP diff / SSP step consistency (std) ")
    ax2.set_ylabel("Normalised pBSP step consistency (std) ")
    ax2.set_xlabel("Worker number")
    ax2.legend()

    plt.grid(linestyle='--', linewidth=1)
    plt.show()



def exp_dummy(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(3), 'pbsp_p3'), (pssp(4, 3), 'pssp_s4_p3')
    ]
    observe_points = []
    configs = [
        {'stop_time':10, 'size':9, 'straggler_perc':0, 'straggleness':1, 'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}
    ]
    for c in configs: run(c)
