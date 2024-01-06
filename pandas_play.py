import numpy as np
import pandas as pd


class Dog:
    tricks_shared = []  # This is shared by all objects of class Dog. Only one array.

    # constructor of the class
    def __init__(self, tricks):
        self.tricks = tricks

    # function of the class.
    def add_trick(self, trick):
        self.tricks.append(trick)
        print("This is the Parent class")


# Inheritance

class Mojo(Dog):

    def __init__(self, tricks):
        print("Child class constructor is called")
        super().__init__(tricks)
        self.index = len(tricks)

    def add_trick(self, trick):
        super().add_trick(trick)
        print("This is the child class")

    # This is also valid
    def add_trick_parent(self, trick):
        Dog.add_trick(self, trick)

    # Make this class iterable
    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.tricks[self.index]

    # Operator overloading. This is what happens underneath in a Series or NumPy object
    def __add__(self, other):
        if isinstance(other, Mojo):
            return Mojo(self.tricks + other.tricks)
        elif isinstance(other, int):
            return Mojo(self.tricks.append(other))
        else:
            raise TypeError("unsupported operand type(s) for +")


# Pandas data frame - 2D Array. Where each column is a Pandas series. Pandas series - They are used to store to one
# dimensional data. The underlying data structure for a pandas.Series is a NumPy ndarray. A pandas.Series can be
# thought of as a labeled, one-dimensional array that provides additional functionality and abstractions built on top
# of the NumPy ndarray structure

sampleDictionary = {'a': 1, 'b': 2, 'c': 3}
# construct a series from a dict
pdSeries: pd.Series = pd.Series(data=sampleDictionary)

# Creating a series from a numpy array

npArray = np.array([1, 2])

#reshape in numpy

y = np.array([1, 2, 3, 4, 5,6])
print(y.reshape(3,-1)) # -1 basically states any argument that needs to make this in to an array of 3 rows


pdNpSeries = pd.Series(npArray, ['a', 'b'])

# Conditionals to filter certain rows and columns in a Data frame

# Using conditional
pdDataFrame = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [True, False, False],
    'c': ['Sharan', 'Prasad', 'IDont']
})

# This produces a series which only contains true or false. The generated series also contains the name attribute.
filterSeries = pdDataFrame['a'] > 1

# Now this can be used to filter the dataFrame
above1DataFrame = pdDataFrame[filterSeries]

# Using for loops for series. This produces an array first.
arrayFirst = [val == 'Sharan' for val in pdDataFrame.c]
# print(arrayFirst)
# print(pdDataFrame[arrayFirst])

# Reindexing - Important for shuffling training and test data.
sampleData = {
    "age": [50, 40, 30, 40],
    "qualified": [True, False, False, False]
}
dfToBeReindexed = pd.DataFrame(sampleData)
print(dfToBeReindexed)
newdf = dfToBeReindexed.reindex(np.random.permutation(dfToBeReindexed.index))
print(newdf)


