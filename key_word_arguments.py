import numpy as np


# Functions take two kind of arguments, positional and keyword arguments.


# For a normal function we can specifiy both positional or keyword arguments
def add(a, b):
    return a + b


# Here first we need to specify positional arguments and then we can specify keyword arguments.
add(1, b=2)
add(a=1, b=2)
add(1, 2)


# Keyword only function. Anything specified after a * needs to be passed as keyword arguments only.
def add_k_only(*, a, b):
    return a + b


# add_k_only(1, 2) ->  will throw an error.
add_k_only(a=1, b=2)


# Take positional arguments only. All the parameters before / can only be positional arguments. In this case all args.
def add_p_only(a, b, /):
    return a + b


add_p_only(1, 2)


# add_p_only(1, b=2) will throw an error.


# You can create a function that accepts any number of positional arguments as well as some keyword-only arguments by
# using the * operator to capture all the positional arguments and then specify optional keyword-only arguments after
# the * capture

def product(*numbers, initial=1):
    total = initial
    for n in numbers:
        total *= n
    return total


product(1, 2, 3, 5, 6, initial=2)


# Similar to the above function we can also specify an arbitrary number of keyword arguments as well.

# Capturing arbitrary keyword arguments

def format_attributes(**attributes):
    """Return a string of comma-separated key-value pairs."""
    return ", ".join(
        f"{param}: {value}"
        for param, value in attributes.items()
    )


format_attributes(name='Sharan', some='Random')

# Other examples -
print(np.array([1, 2, 3]).shape)
print(np.reshape(np.array([[1, 2, ], [3, 4]]), newshape=(2, -1)).shape)
