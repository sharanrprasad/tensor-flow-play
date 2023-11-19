import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = 1 + (1 / 10000) - (1 - 1 / 10000)
print("output : %.18f" % x)

print("actual output should have been: %.18f" % (2 / 10000))

# The above code is not equal actually equal to 2/10000 even though mathematically they are. This is because of how
# large floating point are handled in binary form. This is the reason we sometimes don't specify softmax on the
# output layer but in cost function in tensor flow.


# Building softmax model using the most straight forward approach.
# Because of the problem stated above this might not give the best results.
# It is better use softmax inside the cost function.
model = Sequential(
    [
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(4, activation='softmax')  # < softmax activation here, output layer. 
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),

)

# Inside cost function

preferred_model = Sequential(
    [
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(4, activation='linear')  # <-- Linear is equal to no activation function. That Ai = Zi
    ]
)

preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # <-- This uses softmax in the cost function. This cost function is -log
    optimizer=tf.keras.optimizers.Adam(0.001),  # <-- This is on top of gradient descent. ADAM refers to adaptive
    # learning. It automatically adjusts learning rate. 0001 is starting learning rate.
)


def mysoftmax(z):
    """ Softmax converts a vector of values to a probability distribution.
     Args:
       z (ndarray (N,))  : input data, N features
     Returns:
       a (ndarray (N,))  : softmax of z
     """
    total_exp = np.sum(np.exp(z))

    def exp_divide(x):
        return np.exp(x) / total_exp

    return np.vectorize(exp_divide)(z)


print(mysoftmax(np.array([1, 2, 3])))
