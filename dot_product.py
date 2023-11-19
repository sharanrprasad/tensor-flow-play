# Vector vs matrices ->
# In summary, a vector is a one-dimensional array representing
# a quantity with magnitude and direction, while a matrix is a two-dimensional array
# representing a collection of numbers arranged in rows and columns. Vectors are often
# used to represent individual data points or
# quantities, while matrices are used to represent collections or structured datasets.

# Vectors are usually represented by a column lke the one below
# [
# 2
# 3
# ]
# Transpose of the above vector becomes - [2,3]

# All the steps that were defined in forward-prop dense function can be replaced by np.matmul method which
# basically does matrix multiplication.


# One important point about matrix multiplication is that the dot product of two vectors is also equal to the
# transpose of one vector multiplied by the other

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid_vector = np.vectorize(sigmoid)


# Same function as in forward-prop.py but done using matrix multiplication.
def dense(a_in, W, b):
    z = np.add(np.matmul(a_in, W), b)
    a_out = sigmoid_vector(z)
    return a_out

print(dense([[1, 2]], [[2, 3, 4], [5, 6, 7]], [[2, 3, 6]]))


def mse(y, yhat) :
    """
   Calculate the mean squared error on a data set.
   Args:
     y    : (ndarray  Shape (m,) or (m,1))  target value of each example
     yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
   Returns:
     err: (scalar)
   """
    m = len(y)
    err = 0.0
    for i in range(m):
        ### START CODE HERE ###
        err_i = (y[i] - yhat[i])**2
        err = err + err_i
    err = err / (2*m)
    ### END CODE HERE ###
    return(err)