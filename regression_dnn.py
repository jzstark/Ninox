from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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
epochs=2
batch_size=64
iteration=10
data_select_step = 50
learning_rate=0.001

"""
Load and pre-process MNSIT data
"""

img_rows, img_cols = 28, 28
num_classes = 10

(x_train, y_train),(x_test, y_test) = load_data()
# Sort dataset
idx = np.argsort(y_train)
x_train = x_train[idx]
y_train = y_train[idx]
idx = np.argsort(y_test)
x_test = x_test[idx]
y_test = y_test[idx]

x_train    = x_train.astype('float32') / 255.0
x_test     = x_test.astype('float32') / 255.0
train_len  = len(y_train)
test_len   = len(y_test)
x_train    = np.reshape(x_train, (train_len, img_rows, img_cols, 1))
x_test     = np.reshape(x_test, (test_len, img_rows, img_cols, 1))
y_train    = to_categorical(y_train, num_classes)
y_test     = to_categorical(y_test, num_classes)
x_test_small = x_test[:5000]
y_test_small = y_test[:5000]


"""
Utilities
"""

def get_next_batch(i, n, clock):
    size = batch_size * iteration
    slicelen = int((train_len - size - 1) / n)
    idx = i * slicelen + (clock * data_select_step) % slicelen
    return x_train[[idx,idx+size], :], y_train[[idx,idx+size], :]

def dnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

common_model = dnn()

def diff_weight(w_new, w_old):
    ws = []
    for w, u in zip(w_new, w_old):
        ws.append(w - u)
    return ws

"""
Exposed API for simulation use
"""

def make_optimiser():
    # This has to be newly created for each new instance.
    return optimizers.SGD(lr=learning_rate)
    #return optimizers.Adadelta()

def get_weight(model):
    return model.get_weights()

def set_weight(model, u):
    model.set_weights(u)


def build_model(opt, accuracy=True):
    model = dnn()
    if accuracy == True:
        model.compile(optimizer=opt,
            loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, loss='categorical_crossentropy')

    #np.random.seed(seed)
    #ws = common_model.get_weights()
    #init_weight = [np.random.rand(*w.shape) for w in ws]
    #set_weight(model, init_weight)
    return model


def build_update():
    ws = common_model.get_weights()
    param = [np.zeros(w.shape, dtype='float32') for w in ws]
    return param


def update_model(model, us):
    ws = get_weight(model)
    w_new = []
    for w, u in zip(ws, us):
        w_new.append(w + u)
    set_weight(model, w_new)


# Model, workder id, total worker number,
def compute_updates(model, i, n, step):
    x, y = get_next_batch(i, n, step)
    ws0 = get_weight(model)
    #rint(ws0[0])
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1,
        validation_data=(x_test_small, y_test_small))
    ws1 = get_weight(model)

    set_weight(model, ws0)

    #print(ws1[0])
    #print(diff_weight(ws1, ws0))
    return diff_weight(ws1, ws0)


def compute_accuracy(model):
    loss, accuracy = model.evaluate(x_test_small, y_test_small)
    return loss, accuracy


def test_run():
    N = 5
    opt = make_optimiser()
    model = build_model(opt, accuracy=True)
    for i in range(15):
        updates = compute_updates(model, i % N, N, i)
        update_model(model, updates)
        _,  acc = compute_accuracy(model)
        print("Accuracy %.5f" % acc)
