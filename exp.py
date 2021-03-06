from simulator import *
from utils import *
import database as db
import itertools
import seaborn as sns
import scipy

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as mtick

from scipy.stats import norm

#sns.set_palette(sns.color_palette("hls", 12))

font = 16 #'x-large'
params = {'legend.fontsize': font-2,
          #'figure.figsize': (9.5, 6),
         'axes.labelsize': font,
         'axes.titlesize': font,
         'xtick.labelsize':font,
         'ytick.labelsize':font}
pylab.rcParams.update(params)

markers = ['.', '^', 'o', '*', '+']
linestyles = ['-', '--', '-.', ':', '-']

x = list(itertools.product(linestyles, markers))
random.shuffle(x)
linestyles, markers = tuple(zip(*x))

"""
observation point (for each barriers&size&straggler config):
- "step" : final step of all nodes. Format: one line, rows are all nodes.
- "sequence": two rows, each row has about (#nodes * average_steps) elements.
    It is the order that each update is generated.
    First row: id of node;
    second row: the step of that node when this update is generated;
    third row: the time this update is generated

- "frontier" : progress inconsistency
- "ratio" : time usage ratio
- "regression" : the application, includes linear regression, matrix factorization, and DNN.

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
        {'stop_time':200, 'size':200, 'straggler_perc':0, 'straggleness':1, 'barriers':barriers, 'observe_points':observe_points,
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

    font = 20 #'x-large'
    params = {'legend.fontsize': font-2,
              #'figure.figsize': (9.5, 6),
             'axes.labelsize': font,
             'axes.titlesize': font,
             'xtick.labelsize':font,
             'ytick.labelsize':font}
    pylab.rcParams.update(params)

    dashList = [(1,1), (3,2), (1,0), (3,3), (2, 4), (4,2,10,2), (2,1), (1,2)]
    patterns = [ "/" , ".", "\\" , "-", "|", "x", "o", "O",  "+",  "*", "." ]
    fig, ax = plt.subplots(figsize=(12, 4))
    k = 0

    bin = {'asp':24, 'bsp':1, 'ssp_s4':4, 'pbsp_p10':2, 'pssp_s4_p10':16}
    for name in barrier_names:
        x = data[name]
        counts, bins, bars = ax.hist(x, bins=bin[name],
            label=barrier_to_label(name), alpha=0.8,
            #rwidth=1.,
            hatch=patterns[k])
        print(name, bins)
        #d = sorted(data[name])
        #x, counts = np.unique(d, return_counts=True)
        #y = np.multiply(x, counts) / sum(d)
        #print(y)
        #ax.plot(x, y, label=barrier_to_label(name), linewidth=2.5,
        #    linestyle='--', dashes=dashList[k])
        k += 1
    ax.set_ylim([0, 200])
    plt.xlabel("Steps")
    plt.ylabel("Number of nodes")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    font = 20 #'x-large'
    params = {'legend.fontsize': font-2,
              #'figure.figsize': (9.5, 6),
             'axes.labelsize': font,
             'axes.titlesize': font,
             'xtick.labelsize':font,
             'ytick.labelsize':font}
    pylab.rcParams.update(params)

    dashList = [(1,1), (3,2), (1,0), (3,3), (2, 4), (4,2,10,2), (2,1), (1,2)]
    fig, ax = plt.subplots(figsize=(12, 5))
    k = 0
    for name in barrier_names:
        #n, bins, patches = ax.hist(data[name], 500, cumulative=True, histtype='step', label=barrier_to_label(name))
        #patches[0].set_xy(patches[0].get_xy()[:-1])
        x = sorted(data[name])
        y = np.cumsum(x)
        y = y / y[-1]
        ax.plot(x, y, label=barrier_to_label(name), linewidth=3,
            linestyle='--', dashes=dashList[k])
        k += 1
    ax.set_ylim([0, 1])
    plt.xlabel("Steps")
    plt.ylabel("CDF of nodes")
    plt.legend(loc="lower right")
    plt.tight_layout()
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
        ax.plot(x, y, marker=markers[c], label=barrier_to_label(k),
            linestyle=linestyles[c % len(markers)],
            linewidth=2,  markersize=10,)
        c += 1
    plt.legend()
    plt.xlabel("Straggle node percentage")
    plt.ylabel("Normalised average step progress")
    plt.tight_layout()
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
        ax.plot(x, y, marker=markers[c], label=barrier_to_label(k),
            linestyle=linestyles[c % len(markers)],
            linewidth=2,  markersize=10,)
        c += 1
    plt.legend()
    plt.xlabel("Straggleness of the slow nodes")
    plt.ylabel("Normalised average progress")
    plt.tight_layout()
    plt.show()


def exp_scalability_step(result_dir):
    db.init_db(result_dir)

    ssp_name = 'ssp_s4'
    barriers_ssp = [
        (asp, 'asp'),
        (pssp(4, 2), 'pssp_s4_p2'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 20), 'pssp_s4_p20'),
        (pssp(4, 30), 'pssp_s4_p30'),
        (pssp(4, 40), 'pssp_s4_p40'),
        (pssp(4, 50), 'pssp_s4_p50'),
        (pssp(4, 100), 'pssp_s4_p100'),
        #(pssp(4, 200), 'pssp_s4_p200'),
        (ssp(4), ssp_name),
    ]

    barriers_bsp = [
        (asp, 'asp'),
        (pbsp(2),  'pbsp_p2'),
        #(pbsp(5),  'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(20), 'pbsp_p20'),
        #(pbsp(30), 'pbsp_p30'),
        #(pbsp(40), 'pbsp_p40'),
        (pbsp(50), 'pbsp_p50'),
        (pbsp(100), 'pbsp_p100'),
        (bsp, 'bsp'),
    ]

    barriers = barriers_bsp

    observe_points = ['step']
    configs = [
        {'stop_time':100, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':200, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':300, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':400, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':500, 'straggler_perc':0, 'straggleness':1,
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
                s = [int(s) for s in next(reader)]
                step[barrier] = (np.mean(s), np.std(s))
        steps.append(step)

    #print(steps)

    fig, ax = plt.subplots()
    #barrier_names = [s for (_, s) in barriers if s != ssp_name]
    barrier_names = [s for (_, s) in barriers]
    sizes = [c['size'] for c in configs]
    for k, barrier in enumerate(barrier_names):
        label = barrier_to_label(barrier)
        y = []; y_err = []
        for i, c in enumerate(configs):
            #ratio = np.divide(steps[i][barrier][0], steps[i][ssp_name][0])
            mu = steps[i][barrier][0]
            std = steps[i][barrier][1]
            y.append(mu)
            y_err.append(std)
        ax.errorbar(sizes, y, yerr=y_err, marker=markers[k % len(markers)],
            linewidth=2, markersize=10,
            linestyle=linestyles[k % len(markers)], label=label)
    #ax.set_ylabel("Ratio of PSSP step / SSP step progress")
    ax.set_ylabel("Step progress")
    ax.set_xlabel("Number of workers")
    #plt.grid(linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()


"""
Experiment 2: SGD Accuracy
"""

# - Sequence length vs (accuracy compared to BSP); which node *generates* a new update. I expect pBSP and pSSP are bounded, but not ASP. The definition of "difference" should follow that in math proof.
# (Note that sequence length itself is "number of updates")
# - Change straggler percentage of pBSP, pSSP. Redo Evaluation,
# - Change straggler scale. Redo.
# - Chnage x-axis to real time for all the above

#
def exp_regression(result_dir):
    #exp_regression_lda(result_dir)
    #exp_regression_mnist(result_dir)
    exp_regression_mf(result_dir)


def exp_regression_mnist(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (pbsp(4), 'pbsp_p4'),
        (pbsp(8), 'pbsp_p8'),
        (pbsp(16), 'pbsp_p16'),
        (pbsp(32), 'pbsp_p32'),
        (bsp, 'bsp'),
    ]
    observe_points = ['regression']
    config = {'stop_time':100, 'size': 64, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    #run(config)

    clock = {}; iteration = {}; loss = {}
    barrier_names = [s for (_, s) in config['barriers']]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'regression')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            clock[barrier] = [float(s) for s in next(reader)]
            iteration[barrier] = [float(s) for s in next(reader)]
            loss[barrier] = [float(s) for s in next(reader)]

    c = 0
    fig, ax = plt.subplots(figsize=(7, 6))
    for barrier in barrier_names:
        ax.plot(clock[barrier], loss[barrier], label=barrier_to_label(barrier),
            linewidth=2, markersize=10,
            linestyle=linestyles[c], marker=markers[c])
        c += 1
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.xlim([10,80])
    plt.ylim([0.05, 0.65])
    ax.set_xlabel("Number of updates")
    ax.set_ylabel("Accuracy")

    plt.legend()
    plt.show()


def exp_regression_lda(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (bsp, 'bsp'),
        (ssp(3), 'ssp_s3'),
        (pbsp(4), 'pbsp_p4'),
        (pssp(3, 4),  'pssp_s3_p4'),
    ]
    observe_points = ['regression']
    config = {'stop_time':100, 'size': 32, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    #run(config)


    clock = {}; iteration = {}; loss = {}
    barrier_names = [s for (_, s) in config['barriers']]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'regression')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            clock[barrier] = [float(s) for s in next(reader)]
            iteration[barrier] = [float(s) for s in next(reader)]
            loss[barrier] = [float(s) for s in next(reader)]


    # For the LDA exp #
    dashList = [(1,1),(2,1), (1,0),(3,3,2,2),(5,2,10,2)]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) #figsize=(12, 5))
    c = 0
    for barrier in barrier_names:
        ax1.plot(clock[barrier], loss[barrier],
            linewidth=3,
            #linestyle=linestyles[c],
            linestyle='--', dashes=dashList[c],
            label=barrier_to_label(barrier))
        ax2.plot(iteration[barrier], loss[barrier],
            linewidth=3,
            #linestyle=linestyles[c],
            linestyle='--', dashes=dashList[c],
            label=barrier_to_label(barrier))
        c += 1

    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax1.set_xlabel("Simulated time")
    ax1.set_ylabel("Loglikelihood")
    ax1.legend()

    ax2.set_xlabel("Number of updates")
    #ax2.set_ylabel("Accuracy")
    #ax2.set_xlim([0,50])
    ax1.set_ylim([-2530000,-2480000])
    ax2.set_ylim([-2530000,-2480000])
    ax1.set_xlim([50,200])
    ax2.set_xlim([500,3000])

    ax2.legend()
    #plt.grid()
    plt.show()


def exp_regression_mf(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (pbsp(4), 'pbsp_p4'),
        (pbsp(8), 'pbsp_p8'),
        (pbsp(16), 'pbsp_p16'),
        (pbsp(32), 'pbsp_p32'),
        (pbsp(48), 'pbsp_p48'),
        (bsp, 'bsp'),
    ]
    observe_points = ['regression']
    config = {'stop_time':100, 'size': 63,
        'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    #run(config)

    clock = {}; iteration = {}; loss = {}
    barrier_names = [s for (_, s) in config['barriers']]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'regression')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            clock[barrier] = [float(s) for s in next(reader)]
            iteration[barrier] = [float(s) for s in next(reader)]
            loss[barrier] = [float(s) for s in next(reader)]

    c = 0
    fig, ax = plt.subplots(figsize=(8, 5))
    for barrier in barrier_names:
        ax.plot(clock[barrier][1::2], loss[barrier][1::2], label=barrier_to_label(barrier),
            linewidth=2, markersize=10,
            linestyle=linestyles[c], marker=markers[c])
        c += 1

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.xlim([10,80])
    plt.ylim([73000, 89500])
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("Square loss")
    plt.tight_layout()
    plt.legend()
    plt.show()


def exp_straggle_accuracy(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (bsp, 'bsp'),
        (ssp(4), 'ssp_s4'),
        #(pbsp(10), 'pbsp_p10'),
        #(pssp(4, 10), 'pssp_s4_p10')
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5')
    ]
    observe_points = ['regression']
    t = 40
    s = 60 #1000
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
        #{'stop_time':t, 'size':s, 'straggler_perc':25, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        #'path':result_dir},
        #{'stop_time':200, 'size':100, 'straggler_perc':30, 'straggleness':4, 'barriers':barriers, 'observe_points':observe_points,
        #'path':result_dir},
    ]

    #for c in configs: run(c)

    dict_stragglers = {}
    for b in barriers:
        dict_single_straggler = {}
        for c in configs:
            filename = utils.dbfilename(c, b[1], observe_points[0])
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                clock = [float(s) for s in next(reader)]
                iteration = [float(s) for s in next(reader)]
                loss = [float(s) for s in next(reader)]
                #accuracy = loss[-1]
                #dict_single_straggler[c['straggler_perc']] = accuracy
                accuracy = loss[-5:-1]
                dict_single_straggler[c['straggler_perc']] = \
                    (np.mean(accuracy), np.std(accuracy))
        dict_stragglers[b[1]] = dict_single_straggler

    print(dict_stragglers)

    markers = ['.', '^', 'o', '*', '+']
    linestyles = ['-', '--', '-.', ':', '-']

    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        y = list(i.values())
        y1, y2 = zip(*y)
        y = (np.divide(y1, y1[0]) - 1) * 100
        #ax.errorbar(x, y1, yerr=y2, marker=markers[c], label=barrier_to_label(k))
        ax.plot(x, y, marker=markers[c], linestyle=linestyles[c],
            markersize=10, linewidth=2,
            label=barrier_to_label(k))
        c += 1
    plt.legend()
    plt.xlabel("Percentage of slow nodes")
    plt.ylabel("Percentage of accuracy decrease")
    #plt.ylabel("Model accuracy ")
    plt.show()


def exp_scalability(result_dir):
    db.init_db(result_dir)

    ssp_name = 'ssp_s4'
    barriers_ssp = [
        (ssp(4), ssp_name),
        (pssp(4, 2), 'pssp_s4_p2'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 15), 'pssp_s4_p15'),
        (pssp(4, 20), 'pssp_s4_p20'),
    ]
    barriers = [
        (asp, 'asp'),
        (pbsp(2),  'pbsp_p2'),
        (pbsp(5),  'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(20), 'pbsp_p20'),
        (pbsp(30), 'pbsp_p30'),
        (pbsp(40), 'pbsp_p40'),
        (pbsp(50), 'pbsp_p50'),
        (bsp, 'bsp'),
    ]
    observe_points = ['regression']
    # run 300 seconds
    t = 80
    configs = [
        {'stop_time':t, 'size':50, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':150, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':200, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':250, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':t, 'size':300, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    #for c in configs: run(c)

    clocks = []; iterations = []; losses = []
    for c in configs:
        clock = {}; iteration = {}; loss = {}
        barrier_names = [s for (_, s) in c['barriers']]
        for barrier in barrier_names:
            filename = utils.dbfilename(c, barrier, 'regression')
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                clock[barrier] = [float(s) for s in next(reader)]
                iteration[barrier] = [int(s) for s in next(reader)]
                loss[barrier] = [float(s) for s in next(reader)]
        clocks.append(clock)
        iterations.append(iteration)
        losses.append(loss)

    print(losses)

    fig, ax = plt.subplots(figsize=(8, 4))

    barrier_names = [s for (_, s) in barriers]
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
            #r = np.divide(losses[i][barrier][-1], losses[i][ssp_name][-1])
            r = losses[i][barrier][-1]
            y.append(r)
        ax.plot(sizes, y, label=label,
            marker=markers[k%len(markers)],
            linestyle=linestyles[k%len(linestyles)])
    ax.set_ylabel("Ratio of PSSP/ SSP in regression model loss value")
    ax.set_xlabel("Number of nodes")
    plt.grid(linestyle='--', linewidth=1)
    plt.legend()
    plt.show()


"""
Experiment 3: Sequence Inconsistency
"""

def exp_seqdiff(result_dir):
    db.init_db(result_dir)

    barriers = [
        # (bsp, 'bsp'), --> should be all 0 or very close to it at least
        (asp, 'asp'),
        (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
    ]

    size = 100
    # Observe N different points in the whole updates sequence
    N = int(size/3) # a random step
    observe_points = ['sequence']
    config = {'stop_time':100, 'size':size, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    # run(config)

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

    fig, ax = plt.subplots(figsize=(7, 6))
    dashList = [(1,1),(2,1),(2,2),(1,0),(3,3,2,2),(5,2,20,2)]
    barrier_names = result.keys()
    k = 0
    for barrier in barrier_names:
        [y, t] = list(zip(*result[barrier]))
        #y = np.divide(y, t) # not a good metric
        y = np.divide(y, size)
        ax.plot(t[0::2], y[0::2], label=barrier_to_label(barrier),
            linewidth=2,
            linestyle='--', dashes=dashList[k])
            #linestyle=linestyles[k])
        #k = (k + 1) % len(linestyles)
        k = (k + 1) % len(dashList)
    ax.set_xlabel("Sequence length T (P=%d)" % size)
    ax.set_ylabel("Normalised sequence inconsistency")
    plt.legend()
    plt.show()


def exp_straggle_seqdiff(result_dir):
    db.init_db(result_dir)
    barriers = [
        (asp, 'asp'),
        (ssp(4), 'ssp_s4'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pbsp(5), 'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(20), 'pbsp_p20'),
        #(pbsp(30), 'pbsp_p30'),
        #(pbsp(50), 'pbsp_p50'),
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
    time = 100; size = 100
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
        {'stop_time':time, 'size':size, 'straggler_perc':30, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':40, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':50, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':60, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':70, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':80, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':90, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':100, 'straggleness':4,
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
        L = len(nodes)
        diffs = []
        for i in range(8):
            length = L - i * 30
            true_seq  = generate_true_seq(length, n)
            noisy_seq = list(zip(nodes, steps))
            true_set = set(true_seq)
            noisy_set = set(noisy_seq)
            diff_a_b = true_set.difference(noisy_set)
            diff_b_a = noisy_set.difference(true_set)
            diff = len(diff_a_b) + len(diff_b_a)
            diffs.append(diff)
        return (np.mean(diffs), np.std(diffs))

    fig, ax = plt.subplots(figsize=(7, 5))
    c = 0
    for k, i in dict_stragglers.items():
        sizes  = list(i.keys())   # sizes
        nslist = list(i.values()) # (nodes, steps) list
        diffs = list(map(nsdiff, nslist))
        mu, std = tuple(zip(*diffs))
        mu = np.divide(mu, size)
        std = np.divide(std, size)

        ax.errorbar(sizes, mu, yerr=std, label=barrier_to_label(k),
            linewidth=2,  markersize=10,
            linestyle=linestyles[c%len(linestyles)],
            marker=markers[c%len(markers)])
        c = c + 1

    ax.set_xlabel("Straggler percentage")
    ax.set_ylabel("Normalised sequence inconsistency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def exp_straggleness_seqdiff(result_dir):
    db.init_db(result_dir)
    barriers = [
        (asp, 'asp'), (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10'),
    ]
    barriers = [
        (asp, 'asp'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 20), 'pssp_s4_p20'),
        (pssp(4, 30), 'pssp_s4_p30'),
        (pssp(4, 40), 'pssp_s4_p40'),
        #(pssp(4, 50), 'pssp_s4_p50'),
        #(pssp(4, 70), 'pssp_s4_p70'),
        #(pssp(4, 80), 'pssp_s4_p80'),
        (ssp(4), 'ssp_s4'),
    ]

    """
    barriers = [
        (asp, 'asp'),
        (pbsp(5), 'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(15), 'pbsp_p15'),
        (pbsp(20), 'pbsp_p20'),
        (pbsp(30), 'pbsp_p30'),
    ]
    """

    def generate_true_seq(l, n):
        seq = [None] * l
        for i in range(l):
            c = int(i / n)
            p = i % n
            seq[i] = (p, c+1)
        return seq

    observe_points = ['sequence']
    time = 100; size = 100

    configs = [
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':2,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':4,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':6,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':8,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':10,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':12,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':14,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':size, 'straggler_perc':10, 'straggleness':16,
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
                dict_single_straggler[c['straggleness']] = (nodes, steps, n)
        dict_stragglers[barrier] = dict_single_straggler

    def nsdiff(ns):
        nodes, steps, n = ns
        L = len(nodes)
        diffs = []
        for i in range(5):
            length = L - i * 30
            true_seq  = generate_true_seq(length, n)
            noisy_seq = list(zip(nodes, steps))
            true_set = set(true_seq)
            noisy_set = set(noisy_seq)
            diff_a_b = true_set.difference(noisy_set)
            diff_b_a = noisy_set.difference(true_set)
            diff = len(diff_a_b) + len(diff_b_a)
            diffs.append(diff)
        #return diff
        return (np.mean(diffs), np.std(diffs))

    fig, ax = plt.subplots(figsize=(7,6))
    c = 0
    for k, i in dict_stragglers.items():
        sizes  = list(i.keys())   # sizes
        nslist = list(i.values()) # (nodes, steps) list
        diffs = list(map(nsdiff, nslist))

        #diffs = np.divide(diffs, size)
        mu, std = tuple(zip(*diffs))
        mu = np.divide(mu, size)
        std = np.divide(std, size)

        #ax.plot(sizes, diffs, label=barrier_to_label(k),
        ax.errorbar(sizes, mu, yerr=std, label=barrier_to_label(k),
            linewidth=2,  markersize=10,
            linestyle=linestyles[c%len(linestyles)],
            marker=markers[c%len(markers)])
        c += 1

    ax.set_xlabel("Straggleness")
    ax.set_ylabel("Normalised sequence inconsistency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def exp_scalability_seqdiff(result_dir):
    db.init_db(result_dir)
    barriers = [
        (asp, 'asp'), (ssp(4), 'ssp_s4'),
        (pbsp(5), 'pbsp_p5'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pssp(4, 10), 'pssp_s4_p10'),
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
    time = 100; size = 100
    configs = [
        {'stop_time':time, 'size':50, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':150, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':200, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':250, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':300, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':350, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':400, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':450, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':time, 'size':500, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
    ]

    # for c in configs: run(c)

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

    def nsdiff(ns, tail=0):
        nodes, steps, n = ns
        length = len(nodes)
        if (tail > 0):
            nodes  = nodes[0:-tail]
            steps  = steps[0:-tail]
            length = length - tail

        true_seq  = generate_true_seq(length, n)
        noisy_seq = list(zip(nodes, steps))
        true_set = set(true_seq)
        noisy_set = set(noisy_seq)
        diff_a_b = true_set.difference(noisy_set)
        diff_b_a = noisy_set.difference(true_set)
        diff = len(diff_a_b) + len(diff_b_a)
        # return diff / length
        return diff

    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        sizes  = list(i.keys())   # sizes
        print(sizes)
        nslist = list(i.values()) # (nodes, steps) list
        diffs = list(map(nsdiff, nslist))
        diffs = np.divide(diffs, sizes)
        ax.plot(sizes, diffs, label=barrier_to_label(k),
            linewidth=2,  markersize=10,
            marker=markers[c], linestyle=linestyles[c])
        c = (c + 1) % len(markers)

    """ --> the error bar using last 100 observations is really not obvious; if anything, we need the error bar between multiple runs

    tail = 100
    counter = 0
    fig, ax = plt.subplots()
    c = 0
    for k, i in dict_stragglers.items():
        sizes  = list(i.keys())   # sizes
        nslist = list(i.values()) # (nodes, steps) list

        diffs = []
        for l, ns in enumerate(nslist):
            tail_diffs = []
            for j in range(0, tail):
                tail_diffs.append(nsdiff(ns, tail=j) / sizes[l])
            mu = np.mean(tail_diffs)
            std = np.std(tail_diffs)
            diffs.append((mu, std))
        print(diffs)

        result = list(zip(*diffs))
        mus = result[0]; stds = result[1]
        ax.errorbar(sizes, mus, yerr=stds, label=k, marker=markers[c], linestyle=linestyles[c])
        c += 1
    """

    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Normalised sequence inconsistency")
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
Experiment 4: Progress Inconsistency
"""

def exp_frontier(result_dir):
    import scipy.stats as stats

    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (pbsp(4),  'pbsp_p4'),
        (pbsp(16),  'pbsp_p16'),
        (pbsp(32),  'pbsp_p32'),
        (pbsp(64),  'pbsp_p64'),
        (pbsp(95), 'pbsp_p95'),
        (pbsp(99),  'pbsp_p99'),

        #(ssp(1), 'ssp_s1'),
        #(ssp(5), 'ssp_s5'),
        #(ssp(10), 'ssp_s10'),
        #(ssp(20), 'ssp_s20'),

        #(pbsp(1),  'pbsp_p1'),
        #(pbsp(5),  'pbsp_p5'),
        #(pbsp(10), 'pbsp_p10'),
        #(pbsp(20), 'pbsp_p20'),
        #(pbsp(50), 'pbsp_p50'),
        #(pbsp(80), 'pbsp_p80'),
        #(pbsp(95), 'pbsp_p95'),
        #(pbsp(99), 'pbsp_p99'),

        #(pssp(4, 1), 'pssp_s4_p1'),
        #(pssp(4, 5), 'pssp_s4_p5'),
        #(pssp(4, 10), 'pssp_s4_p10'),
        #(pssp(4, 20), 'pssp_s4_p20'),
        #(pssp(4, 50), 'pssp_s4_p50'),
        #(pssp(4, 90), 'pssp_s4_p90'),
        #(pssp(4, 95), 'pssp_s4_p95'),
        #(pssp(4, 99), 'pssp_s4_p99'),
        #(ssp(4), 'ssp_s4'),

    ]
    observe_points = ['frontier']
    config = {'stop_time':60, 'size':100, 'straggler_perc':0, 'straggleness':1.,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir}

    #run(config)

    diff_num = {}; diff_max = {}; diff_min = {}
    barrier_names = [s for (_, s) in config['barriers'] ]
    for barrier in barrier_names:
        filename = utils.dbfilename(config, barrier, 'frontier')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            diff_num[barrier] = [float(s) for s in next(reader)]
            diff_max[barrier] = [float(s) for s in next(reader)]
            if (barrier != 'pbsp_p95'): # TEMP Hack!!!!
                diff_min[barrier] = [float(s) for s in next(reader)]

    print(diff_num)

    dashList = [(1,1), (3,2), (3,3), (2, 4),(1,0), (4,2,10,2), (2,1), (1,2)]
    fig, ax = plt.subplots(figsize=(8, 3))
    c = 0
    barriers = [(k, v) for (k, v) in diff_num.items() if k != "bsp"]
    for k, v in barriers:
        print(k, np.mean(v))
        v = np.divide(v, config['size'])
        density = stats.gaussian_kde(v)
        x = np.linspace(0, 5, 250)
        #n, x, _ = ax.hist(v, 200, histtype='step', cumulative=False, label=k)
        ax.plot(x, density(x),
            #linestyle=linestyles[c % len(linestyles)],
            linestyle='--', linewidth=2,
            dashes=dashList[c],
            label=barrier_to_label(k))
        c += 1
    #ax.axvline(x=1, linestyle=linestyles[c], label='bsp', color='m')
    ax.legend()
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 2])
    ax.set_xlabel('Normalised progress inconsistency')
    ax.set_ylabel('Density')
    plt.tight_layout()
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
    sns.set_palette(sns.color_palette("hls", 10))

    barriers_bsp = [
        (asp, 'asp'),
        #(pbsp(2), 'pbsp_p2'),
        (pbsp(5), 'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(20), 'pbsp_p20'),
        #(pbsp(30), 'pbsp_p30'),
        (pbsp(40), 'pbsp_p40'),
        #(pbsp(50), 'pbsp_p50'),
        #(pbsp(60), 'pbsp_p60'),
        #(pbsp(70), 'pbsp_p70'),
        (pbsp(80), 'pbsp_p80'),
        #(pbsp(90), 'pbsp_p90'),
        (pbsp(95), 'pbsp_p95'),
        #(pbsp(100), 'pbsp_p100'),
        (bsp, 'bsp'),
    ]

    barriers_ssp = [
        (asp, 'asp'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 20), 'pssp_s4_p20'),
        (pssp(4, 40), 'pssp_s4_p40'),
        #(pssp(4, 60), 'pssp_s4_p60'),
        (pssp(4, 80), 'pssp_s4_p80'),
        #(pssp(4, 90), 'pssp_s4_p90'),
        (pssp(4, 95), 'pssp_s4_p95'),
        #(pssp(4, 99), 'pssp_s4_p99'),
        (ssp(4), 'ssp_s4'),
    ]

    barriers = barriers_ssp

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
        {'stop_time':t, 'size':s, 'straggler_perc':30, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
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
                skew = scipy.stats.skew(diff_num)
                kurt = scipy.stats.kurtosis (diff_num)
                dict_single_straggler[c['straggler_perc']] = (mu, std, skew, kurt)
        dict_stragglers[b[1]] = dict_single_straggler

    # print(dict_stragglers)


    # Mean and Std
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5)) #figsize=(12, 5))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        mu, std, _, _  = zip(*ys)
        #mu = np.divide(mu, mu[0])
        #std = np.divide(std, std[0])
        mu = np.divide(mu, s)
        std = np.divide(std, s)
        ax1.plot(x, mu, marker=markers[c], linestyle=linestyles[c],
            linewidth=2,  markersize=10,
            label=barrier_to_label(k))
        ax2.plot(x, std, marker=markers[c], linestyle=linestyles[c],
            linewidth=2,  markersize=10,
            label=barrier_to_label(k))
        c = (c + 1) % (len(markers))

    ax1.set_xlabel("Straggle node percentage")
    ax1.set_ylabel("Progress inconsistency (mean)")
    ax1.legend()

    ax2.set_xlabel("Straggle node percentage")
    ax2.set_ylabel("Progress inconsistency (stdev)")
    plt.tight_layout()
    plt.show()


    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) #figsize=(12, 5))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        _, _, skew, kurt  = zip(*ys)
        ax1.plot(x, skew, marker=markers[c], linestyle=linestyles[c], label=barrier_to_label(k))
        ax2.plot(x, kurt, marker=markers[c], linestyle=linestyles[c], label=barrier_to_label(k))
        c = (c + 1) % (len(markers))

    ax1.set_xlabel("Straggle node percentage")
    ax1.set_ylabel("Progress inconsistency (Skewness)")
    #ax1.legend(loc='upper right')

    ax2.set_xlabel("Straggle node percentage")
    ax2.set_ylabel("Progress inconsistency (Kurtosis)")
    ax2.legend()
    plt.show()


    f, ax1 = plt.subplots(1, 1, figsize=(6, 12)) #figsize=(12, 5))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        _, _, skew, kurt  = zip(*ys)
        ax1.plot(x, skew, marker=markers[c], linestyle=linestyles[c], label=barrier_to_label(k))
        c = (c + 1) % (len(markers))

    ax1.set_xlabel("Straggle node percentage")
    ax1.set_ylabel("Progress inconsistency (Skewness)")
    ax1.legend(loc='upper left')
    ax1.legend()
    plt.show()
    """


def exp_straggleness_consistency(result_dir):
    db.init_db(result_dir)

    barriers = [
        (asp, 'asp'),
        (pbsp(2), 'pbsp_p2'),
        (pbsp(5), 'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(20), 'pbsp_p20'),
        (pbsp(30), 'pbsp_p30'),
        (pbsp(40), 'pbsp_p40'),
        (pbsp(50), 'pbsp_p50'),
        (pbsp(60), 'pbsp_p60'),
        (pbsp(70), 'pbsp_p70'),
        (pbsp(80), 'pbsp_p80'),
        (pbsp(90), 'pbsp_p90'),
        #(pbsp(95), 'pbsp_p95'),
        #(pbsp(100), 'pbsp_p100'),
        (bsp, 'bsp'),

        #(asp, 'asp'),
        #(pssp(4, 5), 'pssp_s4_p5'),
        #(pssp(4, 10), 'pssp_s4_p10'),
        #(pssp(4, 20), 'pssp_s4_p20'),
        #(pssp(4, 40), 'pssp_s4_p40'),
        #(pssp(4, 60), 'pssp_s4_p60'),
        #(pssp(4, 80), 'pssp_s4_p80'),
        #(pssp(4, 90), 'pssp_s4_p90'),
        #(pssp(4, 95), 'pssp_s4_p95'),
        #(pssp(4, 99), 'pssp_s4_p99'),
        #(ssp(4), 'ssp_s4'),
    ]
    ob = ['frontier']
    t = 100
    s = 100
    configs = [
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':1, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':2, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':4, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':6, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':8, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':10, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':12, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':14, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
        {'stop_time':t, 'size':s, 'straggler_perc':10, 'straggleness':16, 'barriers':barriers, 'observe_points':ob, 'path':result_dir},
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
                skew = scipy.stats.skew(diff_num)
                kurt = scipy.stats.kurtosis (diff_num)
                dict_single_straggler[c['straggleness']] = (mu, std, skew, kurt)
        dict_stragglers[b[1]] = dict_single_straggler

    # print(dict_stragglers)

    # Mean/Std
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        mu, std, _, _ = zip(*ys)
        ax1.plot(x, mu, marker=markers[c], linestyle=linestyles[c], label=barrier_to_label(k))
        ax2.plot(x, std, marker=markers[c], linestyle=linestyles[c], label=barrier_to_label(k))
        c = (c + 1) % (len(markers))

    ax1.set_xlabel("Straggleness")
    ax1.set_ylabel("Average of progress inconsistency")
    #ax1.legend()

    ax2.set_xlabel("Straggleness")
    ax2.set_ylabel("Stdev of progress inconsistency")
    ax2.legend(loc='lower right')
    plt.show()

    """
    # Skewness / Kurtosis
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    c = 0
    for k, i in dict_stragglers.items():
        x = list(i.keys())
        ys = list(i.values())
        _, _, skew, kurt = zip(*ys)
        ax1.plot(x, skew, marker=markers[c], linestyle=linestyles[c], label=barrier_to_label(k))
        ax2.plot(x, kurt, marker=markers[c], linestyle=linestyles[c], label=barrier_to_label(k))
        c = (c + 1) % (len(markers))

    ax1.set_xlabel("Straggleness")
    ax1.set_ylabel("Progress inconsistency (Skewness)")
    #ax1.legend()

    ax2.set_xlabel("Straggleness")
    ax2.set_ylabel("Progress inconsistency (Kurtosis)")
    ax2.legend()
    plt.show()
    """


def exp_scalability_consistency(result_dir):
    db.init_db(result_dir)

    ssp_name = 'ssp_s4'
    barriers = [
        (asp, 'asp'),
        (pssp(4, 2), 'pssp_s4_p2'),
        (pssp(4, 5), 'pssp_s4_p5'),
        (pssp(4, 10), 'pssp_s4_p10'),
        (pssp(4, 20), 'pssp_s4_p20'),
        #(pssp(4, 30), 'pssp_s4_p30'),
        (pssp(4, 40), 'pssp_s4_p40'),
        #(pssp(4, 50), 'pssp_s4_p50'),
        (ssp(4), ssp_name),
    ]
    barriers_bsp = [
        (asp, 'asp'),
        (pbsp(2),  'pbsp_p2'),
        (pbsp(5),  'pbsp_p5'),
        (pbsp(10), 'pbsp_p10'),
        (pbsp(20), 'pbsp_p20'),
        #(pbsp(30), 'pbsp_p30'),
        (pbsp(40), 'pbsp_p40'),
        #(pbsp(50), 'pbsp_p50'),
        #(pbsp(100), 'pbsp_p100'),
        #(pbsp(200), 'pbsp_p200'),
    ]
    observe_points = ['frontier']
    configs = [
        {'stop_time':100, 'size':50, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':100, 'straggler_perc':0, 'straggleness':1,
        'barriers':barriers, 'observe_points':observe_points,
        'path':result_dir},
        {'stop_time':100, 'size':150, 'straggler_perc':0, 'straggleness':1,
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
                skew = scipy.stats.skew(diff_num)
                kurt = scipy.stats.kurtosis (diff_num)
                diff[barrier] = (mu, std, skew, kurt)
        diffs.append(diff)

    # print(diffs)

    # Mean/Std
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6)) #figsize=(12, 5))
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
            linewidth=2,  markersize=10,
            linestyle=linestyles[k % len(markers)], label=label)

    for k, barrier in enumerate(barrier_names):
        label = barrier_to_label(barrier)
        y = []
        for i, c in enumerate(configs):
            #ratio = np.divide(diffs[i][barrier][1], diffs[i][ssp_name][1])
            ratio = np.divide(diffs[i][barrier][1], c['size'])
            y.append(ratio)
        ax2.plot(sizes, y, marker=markers[k % len(markers)],
            linewidth=2, markersize=10,
            linestyle=linestyles[k % len(markers)], label=label)

    #ax1.set_ylabel("Ratio of PSSP diff / SSP step consistency (mean) ")
    #ax1.set_ylabel("Normalised progress inconsistency (mean) ")
    ax1.set_ylabel("Average of progress inconsistency")
    ax1.set_xlabel("Number of workers")
    ax1.legend()

    #ax2.set_ylabel("Ratio of PSSP diff / SSP step consistency (std) ")
    #ax2.set_ylabel("Normalised progress inconsistency (stdev) ")
    ax2.set_ylabel("Stdev of progress inconsistency")
    ax2.set_xlabel("Number of workers")
    ax2.legend()

    #plt.grid(linestyle='--', linewidth=1)
    plt.show()

    """
    # Skewness and Kurtosis
    f, ax1 = plt.subplots(1, 1, figsize=(6,10))
    barrier_names = [s for (_, s) in barriers] #if s != ssp_name]
    sizes = [c['size'] for c in configs]

    for k, barrier in enumerate(barrier_names):
        label = barrier_to_label(barrier)
        y = []
        for i, c in enumerate(configs):
            #ratio = np.divide(diffs[i][barrier][2], c['size'])
            ratio = diffs[i][barrier][2]
            y.append(ratio)
        ax1.plot(sizes, y, marker=markers[k % len(markers)],
            linestyle=linestyles[k % len(markers)], label=label)

    for k, barrier in enumerate(barrier_names):
        label = barrier_to_label(barrier)
        y = []
        for i, c in enumerate(configs):
            ratio = np.divide(diffs[i][barrier][3], c['size'])
            y.append(ratio)
        ax2.plot(sizes, y, marker=markers[k % len(markers)],
            linestyle=linestyles[k % len(markers)], label=label)
    ax1.set_ylabel("pBSP step consistency (skewness) ")
    ax1.set_xlabel("Number of workers")
    ax1.legend()

    plt.grid(linestyle='--', linewidth=1)
    plt.show()
    """


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
Other
"""

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
