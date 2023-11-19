# Tensor flow has automatic differentiation tool which can be very useful
# when calculating gradient descent of a cost function which doesn't fit very well in to neural network model.

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

w = tf.Variable(2.0)
b = tf.Variable(1.0)
x = 1.0
y = 100.0  # Target variable

learning_rate = tf.constant(0.15)

wValues = tf.constant([])
bValues = tf.constant([])

for i in range(100):
    with tf.GradientTape() as tape:
        fwb = w * x + b
        costJ = ((fwb - y) ** 2) / 2
    if costJ <= 0:
        continue
    [diff_dw, diff_db] = tape.gradient(costJ, [w, b])
    w.assign_add(-learning_rate * diff_dw)
    b.assign_add(-learning_rate * diff_db)
    wValues = tf.concat([wValues, [w.value()]], axis=0)
    bValues = tf.concat([bValues, [b.value()]], axis=0)

print(w.numpy(), b.numpy())

fig, ax1 = plt.subplots(1, 1)
ax1.bar(wValues.numpy(), bValues.numpy()) # This is a horrible graph. Need to look up how to plot two different
# arrays properly.

plt.show()

print(wValues.numpy(), bValues.numpy())
