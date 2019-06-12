from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.datasets.mnist import load_data
from keras import optimizers

import numpy as np
import random
import platform
import os

# onMacOS: OMP: Error #15: Initializing libiomp5.dylib, but found
# libiomp5.dylib already initialized. -- perhaps caused by multiple MKL libs?
# Another solution could be `conda install nomkl`
#if platform.system() == "Darwin":
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Parameters
"""

seed=233
num_classes = 10
epochs=1
batch_size=512
iteration=5

data_select_step = 100

"""
Load and pre-process MNSIT data
"""

(x_train, y_train),(x_test, y_test) = load_data()
image_size = x_train[0].shape[0] * x_train[0].shape[1]
x_train    = x_train.astype('float32') / 255.0
x_test     = x_test.astype('float32') / 255.0
train_len  = len(y_train)
test_len   = len(y_test)
x_train    = np.reshape(x_train, (train_len, image_size))
x_test     = np.reshape(x_test, (test_len, image_size))
#x_train    = np.reshape(x_train, (train_len, 28, 28, 1))
#x_test     = np.reshape(x_test, (test_len, 28, 28, 1))
y_train    = to_categorical(y_train, num_classes)
y_test     = to_categorical(y_test, num_classes)
x_test_small = x_test[:5000]
y_test_small = y_test[:5000]


"""
Utilities
"""

def make_optimiser():
    # This has to be newly created for each new instance.
    return optimizers.SGD(lr=0.01)


# A worker's data is limited; each has different "local" data.
def get_next_batch(i, n, clock):
    size = batch_size * iteration
    slicelen = int((train_len - size - 1) / n)
    idx = i * slicelen + (clock * data_select_step) % slicelen
    #idx = random.randint(i * slicelen, i * slicelen + slicelen - 1)
    return x_train[[idx,idx+size], :], y_train[[idx,idx+size], :]


def get_weight(model):
    l = model.layers[0]
    return l.get_weights()

def set_weight(model, u):
    l = model.layers[0]
    [w, b] = u
    # Notice this copy!!!!
    w1 = w.copy(); b1 = b.copy()
    return l.set_weights([w1, b1])


"""
Exposed API for simulation use
"""


def build_model(opt, accuracy=True):
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', input_shape=(image_size,)))
    if accuracy == True:
        model.compile(optimizer=opt,
            loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, loss='categorical_crossentropy')

    np.random.seed(seed)
    w_init = np.random.rand(28*28, 10)
    b_init = np.zeros(10)
    set_weight(model, [w_init, b_init])
    return model


def build_update():
    return [np.zeros((28 * 28, 10)), np.zeros(10)]


def update_model(model, u):
    [w0, b0] = get_weight(model)
    w1 = w0 + u[0]
    b1 = b0 + u[1]
    set_weight(model, [w1, b1])
    return model

# Model, workder id, total worker number,
def compute_updates(model, i, n, step):
    x, y = get_next_batch(i, n, step)
    [w0, b0] = get_weight(model)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1,
        validation_data=(x_test_small, y_test_small))
    [w1, b1] = get_weight(model)
    return (w1 - w0, b1 - b0)


def compute_accuracy(model):
    loss, accuracy = model.evaluate(x_test_small, y_test_small, batch_size=64)
    return loss, accuracy
