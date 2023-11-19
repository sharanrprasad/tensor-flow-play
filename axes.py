# Axis represents the number of dimensions of an array. A 3D array has 3 axes.

# Axis can mean different on each numpy function.

# Many functions in NumPy require that you specify an axis along which to apply a certain calculation.
#
# Typically the following rule of thumb applies:
#
# axis=0: Apply the calculation “column-wise”
# axis=1: Apply the calculation “row-wise”

# It’s best to think about NumPy axes as directions long which we can perform operations.
import numpy as np

matrix = np.array([[[1, 2], [3, 4], [5, 6]],
                   [[1, 2], [3, 4], [5, 6]]])

print(matrix.shape)
# When axis is set to zero then sum is calculated for each column
print(np.sum(matrix, axis=0))
print(np.sum(matrix, axis=0).shape)

# When axis is set to one then sum is calculated across the rows.
print(np.sum(matrix, axis=1))
print(np.sum(matrix, axis=1).shape)

# When axis is set to two then it can get a bit more complicated.
print(np.sum(matrix, axis=2))
print(np.sum(matrix, axis=2).shape)

# What ever axis is specified that gets collapsed in the return. Run the above code and then check the original shape
# and then shape of the result.

# When axis is set to -1 here it is equal to call with axis=2.
print(np.sum(matrix, axis=-1))

# Arg max
print(np.argmax(matrix, axis=2))

