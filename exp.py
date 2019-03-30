from simulator import *
import database as db

import matplotlib.pyplot as plt

total_time =

"""
observation point (for each barriers&size&straggler config):
- "step" : final step of all nodes. Format: one line, rows are all nodes.
- "sequence": two rows, each row has about (#nodes * average_steps) elements.
    It is the order that each update is generated.
    First row: id of node;
    second row: the step of that node when this update is generated;
    third row: the time this update is generated

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
        {'size':100, 'straggler_perc':0, 'straggleness':1, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir}
    ]

    # for c in configs: run(c)

    data = {}
    barrier_names = [s for (_, s) in barriers]
    for name in barrier_names:
        filename = utils.dbfilename(configs[0], name, 'step')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data[name] = [int(s) for s in next(reader)]
    print(data)

    fig, ax = plt.subplots()
    for name in barrier_names:
        ax.hist(data[name], 50, label=name)
    plt.legend()
    # plt.show()

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
        {'size':100, 'straggler_perc':0, 'straggleness':1, 'barriers':barriers, 'observe_points':['step'],
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
    print(data)

    fig, ax = plt.subplots()
    for name in barrier_names:
        ax.hist(data[name], 500, cumulative=True, histtype='step', label=name)
    plt.legend()
    #plt.show()


def exp_straggle_perc(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    observe_points = ['step']
    configs = [
        {'size':100, 'straggler_perc':0, 'straggleness':2, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'size':100, 'straggler_perc':5, 'straggleness':2, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'size':100, 'straggler_perc':10, 'straggleness':2, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'size':100, 'straggler_perc':15, 'straggleness':2, 'barriers':barriers, 'observe_points':['step'],
        'path':result_dir},
        {'size':100, 'straggler_perc':20, 'straggleness':2, 'barriers':barriers, 'observe_points':['step'],
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
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        y, _ = zip(*ys)
        y = np.divide(y, y[0])
        ax.plot(x, y, label=k)
    plt.legend()
    plt.show()


"""
Experiment 2: "Accuracy"
"""

# - Sequence length vs (accuracy compared to BSP); which node *generates* a new update. I expect pBSP and pSSP are bounded, but not ASP. The definition of "difference" should follow that in math proof.
# (Note that sequence length itself is "number of updates")
# - Change straggler percentage of pBSP, pSSP. Redo Evaluation,
# - Change straggler scale. Redo.
# - Chnage x-axis to real time for all the above


def exp_accuracy(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'), (bsp, 'bsp'), (ssp(4), 'ssp_s4'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10')
    ]
    observe_points = ['step']
    configs = [
        {'size':50, 'straggler_perc':0, 'straggleness':1, 'barriers':barriers, 'observe_points':['sequence'],
        'path':result_dir}
    ]

    # for c in configs: run(c)

    nodes = {}; steps = {}; times = {}
    barrier_names = [s for (_, s) in barriers]
    for barrier in barrier_names:
        filename = utils.dbfilename(configs[0], barrier, 'sequence')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            nodes[barrier] = [int(s) for s in next(reader)]
            steps[barrier] = [int(s) for s in next(reader)]
            times[barrier] = [float(s) for s in next(reader)]




"""
Experiment 3: Comparison of time used on running/waiting/transmission.
"""

# - Bar chart. Take only final status (or whole process if you want, but the point is not very clear). Compare barriers.

"""
Experiment 4: Scalability
"""

# Leave this to be decided at this stage. Before using notes as x-axis, I need to do some observation of the Accuracy vs. time/iteratio performance under different network size and **sampling size**. Then I can decide what is a good way to present the scability of sampling size.
