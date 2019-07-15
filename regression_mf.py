import numpy as np
import random
import platform
import os
import csv

"""
Parameters
"""

seed = 233
batch_size = 10000 #3125 #20000
alpha = 0.001
beta  = 0.02
data_select_step = 1000

"""
Load and pre-process Movielens-1M data
"""

""" Dataset: 1M
U = 6040 # Number of users
D = 3952 # Number of items
"""
"""
Dataset: 10M
U = 71567 # Number of users
D = 10681 # Number of items
"""

# Dataset 100K
U = 610
D = 9742

K = 100  # Number of features

def read_csv(filename):
    data = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for r in csv_reader:
            r[0] = int(r[0]); r[1] = int(r[1])
            r[2] = float(r[2])
            data.append(r)
    return data

ratings = read_csv("data/ratings_train.csv")
validation_set = read_csv("data/ratings_test.csv")

random.shuffle(ratings)

# global bias
b = np.mean([r[2] for r in ratings])

"""
Utilities
"""

def make_optimiser():
    return None

train_len = len(ratings)

def get_next_batch(i, n, clock):
    slicelen = int((train_len - batch_size - 1) / n)
    #idx = i * slicelen + (clock * data_select_step) % slicelen
    #idx = random.randint(i * slicelen, i * slicelen + slicelen - 1)
    idx = i * slicelen
    #idx = 0
    return ratings[idx : idx + batch_size - 1]


def get_rating(model, i, j):
    p = model['p']; q = model['q']
    bu = model['bu']; bd = model['bd']
    prediction = b + bu[i] + bd[j] + p[i, :].dot(q[j, :].T)
    return prediction

"""
Exposed API for simulation use
"""

def get_weight(model):
    p_copy  = np.copy(model['p'])
    q_copy  = np.copy(model['q'])
    bu_copy = np.copy(model['bu'])
    bd_copy = np.copy(model['bd'])
    return (p_copy, q_copy, bu_copy, bd_copy)


def set_weight(model, weight):
    model['p']  = weight[0]
    model['q']  = weight[1]
    model['bu'] = weight[2]
    model['bd'] = weight[3]


def build_model(opt, accuracy=True):
    np.random.seed(seed)
    p = np.random.normal(scale=1./K, size=(U, K))
    q = np.random.normal(scale=1./K, size=(D, K))
    bu = np.zeros(U)
    bd = np.zeros(D)
    model = {"p": p, "q": q, "bu": bu, "bd": bd}
    return model


def build_update():
    p  = np.zeros((U, K))
    q  = np.zeros((D, K))
    bu = np.zeros(U)
    bd = np.zeros(D)
    return (p, q, bu, bd)


def update_model(model, update):
    (p0, q0, bu0, bd0) = update
    model['p'] = model['p'] + p0
    model['q'] = model['q'] + q0
    model['bu'] = model['bu'] + bu0
    model['bd'] = model['bd'] + bd0
    return model


# model, worker index, total number of workers
def compute_updates(model, wid, n, step):
    p = model['p']; q = model['q']
    bu = model['bu']; bd = model['bd']

    # Make copy
    p_init = np.copy(p); q_init = np.copy(q)
    bu_init = np.copy(bu); bd_init = np.copy(bd)

    samples = get_next_batch(wid, n, step)
    for i, j, r in samples:
        pred = get_rating(model, i, j) # !!! the model changed
        e = r - pred
        bu[i] += alpha * (e - beta * bu[i])
        bd[j] += alpha * (e - beta * bd[j])
        p[i, :] += alpha * (e * q[j, :] - beta * p[i,:])
        q[j, :] += alpha * (e * p[i, :] - beta * q[j,:])

    updates = (p - p_init, q - q_init, bu - bu_init, bd - bd_init)

    set_weight(model, [p_init, q_init, bu_init, bd_init])
    return updates


def add_weight(ws, us):
    (p0, q0, bu0, bd0) = ws
    (p1, q1, bu1, bd1) = us
    return (p0 + p1, q0 + q1, bu0 + bu1, bd0 + bd1)


def average_update(us):
    return list(functools.reduce(add_weight, us))


def compute_accuracy(model):
    sum_error = 0.
    for i, j, r in validation_set:
         pred = get_rating(model, i, j)
         sum_error += abs(r - pred)
    return None, sum_error


def test_run():
    N = 1000
    opt = make_optimiser()
    model = build_model(opt, accuracy=True)
    for i in range(50):
        updates = compute_updates(model, i%N, N, i)
        update_model(model, updates)
        _,  error = compute_accuracy(model)
        print("Error: %.2f" % error)
