import tensorflow as tf
import numpy as np
import random
mnist = tf.keras.datasets.mnist

step_sz = 0.0001

def onehot(y):
    length = len(y)
    onehot = np.zeros((length, 10))
    onehot[np.arange(length), y] = 1
    return onehot

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

train_data_length = 60000

x_train = np.reshape(x_train, (60000, 28 * 28))
x_test  = np.reshape(x_test, (10000, 28 * 28))

y_train = onehot(y_train)
y_test  = onehot(y_test)

x_test_small = x_test[:1000]
y_test_small = y_test[:1000]


def train_data():
    num = random.randint(0, train_data_length)
    yield x_train[[num], :], y_train[[num], :]


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def loss(model):
    x = x_test_small # 1000 * 784
    y = y_test_small # 1000 * 10
    y0 = np.matmul(x, model) # 1000 * 10
    return cross_entropy(softmax(y0), y)


def numgrad(x, y, model):
    y1 = np.matmul(x, model) # 1 * 10
    x = np.transpose(x)
    return np.matmul(x, (y1 - y)) * step_sz
