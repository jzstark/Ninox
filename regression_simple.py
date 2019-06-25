import numpy as np
import random

np.random.seed(9528)

N = 100

step_sz = 0.00001
batch = 5

data_sz, model_sz = 1000 * N, 100
true_model = np.random.uniform(size=[model_sz, 1])
data = np.random.uniform(size=[data_sz, model_sz])
pred = np.dot(data, true_model)
data = np.random.normal(0, 0.001, [data_sz, model_sz]) + data


def make_optimiser():
    return None

def get_next_batch(i, n, clock):
    slicelen = int((data_sz - batch - 1) / n)
    #idx = random.randint(i * slicelen, i * slicelen + slicelen - 1)
    idx = i * slicelen + clock % slicelen
    return data[idx:idx + batch, :], pred[idx:idx + batch]


def build_model(opt, accuracy=True):
    return np.random.uniform(size=[model_sz, 1])


def build_update():
    return np.zeros((model_sz, 1))


#!!! return !!!
def update_model(model, u):
    return model - u


def compute_updates(model, i, n, step):
    x0, y0 = get_next_batch(i, n, step)
    y1 = np.dot(x0, model)
    return np.dot(x0.T,  (y1 - y0)) * step_sz


def compute_accuracy(model):
    y_pred = np.dot(data, model)
    return None, np.mean((y_pred - pred) ** 2)


def test_run():
    N = 5
    opt = make_optimiser()
    model = build_model(opt)
    for i in range(666):
        updates = compute_updates(model, i % N, N, i)
        model = update_model(model, updates)
        _,  acc = compute_accuracy(model)
        print("Accuracy %.5f" % acc)
