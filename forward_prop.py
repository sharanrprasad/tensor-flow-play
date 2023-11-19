# This tries to build a basic implementation of the neural network forward prop algorithm using numpy
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# a_in is the output from the previous neuron. W is the weights matrix and b is the bias vector.
# number of units is not explicitly defined. we get that from the weights matrix
"""
    param a_in: ndarray(n,) - this how you write tuples
    param W: ndarray(n, j) - j units here. This is kind of counter intuitive to how we generally put data in a matrix. It's the opposite. 
    param b: ndarray(n,)
    return: a_out
"""


def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for value in range(units):
        w = W[:, value]
        z = np.dot(a_in, w) + b[value]
        a_out[value] = sigmoid(z)
    return a_out


def my_sequential(x, W1, b1, W2, b2):
    a1 = dense(x, W1, b1)
    print(a1)
    a2 = dense(a1, W2, b2)
    return a2


# This represents first layer with 3 units and second layer with one unit.
W1_temp = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
b1_temp = np.array([1, 2, 3])

W2_temp = np.array([
    [1],
    [4],
    [6]
])
b2_temp = np.array([1])

# Input has two values per input.
x_temp = np.array([1, 2])


def predict():
    out = my_sequential(x_temp, W1_temp, b1_temp, W2_temp, b2_temp)
    return out


print(predict())
