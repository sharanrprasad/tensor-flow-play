# This is importing a package which is a collection of files. Try and keep module names to a single word.
from moduleplay.operations import add
from moduleplay import operations
# This is importing a module which is just a file.
import pandas_play


print(add(2, 3))
print(operations.add(2,3))
print(pandas_play.Dog)
