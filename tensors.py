import numpy as np
import tensorflow as tf

# All tensors are immutable.

# Tensors can be of any number of axes.

# Rank 0 or scalar tensor -
rank_0 = tf.constant(4)
print(rank_0)

# Vector or rank 1 tensor. It has one axes
rank_1 = tf.constant([1.0, 2.0, 3.0], dtype=float)

print(rank_1)

rank_2_tensor = tf.constant([[1, 2, 3],
                             [3, 4, 6],
                             [5, 6, 8]], dtype=tf.float16)

print(rank_2_tensor)

# Rank here represents the number of axes(or dimensions). It will be easy to think in terms of the number of levels
# we need to access a number. Like a[0][1][2][3] means it has a rank of 4.

# An axis of a tensor is a specific dimension of a tensor. The length of an axis tells us how many indexes are
# available along each axis
rank_4_tensor = tf.constant([
    [
        [
            [77, 72],
            [73, 74]
        ],
        [
            [71, 72],
            [73, 74]
        ]
    ]
])

print(rank_4_tensor[0][0][0][0])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
# Shape represents the number of elements of each of the axes of a tensor.
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# Vector normalization - Note we are only talking about vectors which have only one row.
# That is why axis parameter needs to be given for normalization functions. There are different kinds of normalization
# but the most important is L2 which gives a unit vector or a vector with a magnitude of 1.


# Vector norm - Norm is the way to measure the magnitude of a vector. There are different ways to do this but the
# most common one. This is equal to square root of the vector square and is represented as ||V||
# This can be calculated as below in tensor flow.

print(tf.norm(tf.constant([
    [3.0, 4.0]
], dtype=tf.float32), axis=1, ord='euclidean').numpy())

# For any given vector V and it's norm ||V||, it's unit vector is calculated  as (V / ||V||)
# We can  use the function below to calculate the L2 normalisation directly.
print(tf.linalg.l2_normalize(tf.constant([[3, 4]], dtype=tf.float32), axis=1))


