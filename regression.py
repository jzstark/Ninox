from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

num_classes = 10
epochs=1
batch_size=64
iteration=10

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
x_test_small = x_test[:1000]
y_test_small = y_test[:1000]

def get_next_batch():
    size = batch_size * iteration
    idx = random.randint(0, train_len - size - 1)
    return x_train[[idx,idx+size], :], y_train[[idx,idx+size], :]


def build_model():
    model = Sequential()
    """
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
        padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    """
    model.add(Dense(num_classes, activation='softmax', input_shape=(image_size,)))

    sgd = optimizers.SGD(lr=0.005, decay=1e-4)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_weight(model):
    l = model.layers[0]
    return l.get_weights()

def set_weight(model, w, b):
    l = model.layers[0]
    return l.set_weights([w, b])

def update_model(model, u):
    [w0, b0] = get_weight(model)
    w1 = w0 + u[0]
    b1 = b0 + u[1]
    set_weight(model, w1, b1)
    return model

def compute_updates(model):
    x, y = get_next_batch()
    [w0, b0] = get_weight(model)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0,
        validation_data=(x_test_small, y_test_small))
    [w1, b1] = get_weight(model)
    return (w1 - w0, b1 - b0)

def compute_accuracy(model):
    loss, accuracy = model.evaluate(x_test_small, y_test_small, batch_size=64)
    return loss, accuracy
