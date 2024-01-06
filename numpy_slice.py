import numpy as np

one_dimension = np.array([1, 2, 3, 4, 5, 6])

# We pass slice instead of index like this: [start:end]. end is optional. end number is also not considered.
# End - Say we want to get the till the last element , then it is one more than the length of the array.
# We can also define the step, like this: [start:end:step].

# Simple one dimension slice.
print(one_dimension[2:7])
print(one_dimension[2:7:2])

# Negative slicing
print(one_dimension[-1:])
print(one_dimension[-4:-2])

# Reverse an array
print(one_dimension[::-1])

two_dimension = np.array([[1, 2], [3, 4], [5, 6]])

# Two dimension slicing has a comma in between

# Take all rows but only first column
print(two_dimension[0:, 0])

# Take everything from second row and only second column
print(two_dimension[1:, 1])

# Normal arrays. Most of numpy stuff apply here as well. Just for one dimension array
normal_array = [1, 2, 3, 4, 5, 6]

print(normal_array[1:6:2])

# normal_two_dimension = [[1, 2], [3, 4], [5, 6]]
# print(normal_two_dimension[1:, 1]) - This would throw an error


# Filtering by boolean
test = np.array([[2, 1, -1], [0, 1, 1]])
# print(test[0] > 0) # Provides a boolean array [False, True, False]
# print(test[0, test[0] > 0]) # Returns an array where values are greater than zero in row 0. [2,1]
mask = test > 0

print(np.sum(test * mask, axis=1))  # Sum is calculated for all values. All negative numbers before are zero now.

