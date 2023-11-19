# pyplot is a module which contains simple functions and also has the base subplot function.
from matplotlib import pyplot as plt
import numpy as np


# args here means variable number of function arguments. Just like js.
def convert_args_to_array(*args):
    return args


def get_x_and_y():
    return convert_args_to_array(1, 2, 3), [4, 5, 6]


x, y = get_x_and_y()
# the number of arguments here is not fixed. it takes *args
plt.plot(x, y)
# plt.show()

# Subplots

# Specify the number of rows and columns the subplots must be displayed in. The last arg is the index. This uses
# implicit API. More info below.
plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.subplot(2, 2, 2)
plt.plot(x, y)
plt.subplot(2, 2, 3)
plt.plot(x, y)
plt.subplot(2, 2, 4)
plt.plot(x, y)

# plt.show()

# Figures/Axes/Axis/Artist
# Figure is the canvas that we are drawing on.
# Axes is a region inside figure that contain its own set of Artist. It can have 2 0r 3 Axis.
# Ticks are the markers along x or y axis (like 10, 20 etc)


# pyplot(implicit) vs explicit approach. pyplot is like a layer on top and is useful for quick solutions. If we use
# figures and axes directly then we can use that to make all kind of fine changes.

# This contains 4 axes and one figure. uses explict API. Uses list unpacking to set the variables ax1. Note this is
# not a tuple but just unpacked variables.
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))
fig.set_facecolor("red")

ax1.set_facecolor("blue")
ax1.plot(x, y)
ax2.set_facecolor("green")
ax3.set_facecolor("yellow")
ax3.plot(x, y)
ax4.set_facecolor("violet")
ax4.plot(x, y)
ax4.plot(x, x)

# axis.plot creates Artists(includes things like label, line, legends etc.). They can further be styled

l, = ax4.plot(x, y)
l.set_linestyle(':')
ax4.text(1, 4, r'$\mu=115,\ \sigma=15$')
ax4.grid(True)
# Set markets along y-axis
ax4.set_yticks([-1.5, 0, 1.5])
# This graph shows how figures and axes are different.


# Bar graph
fig_bar, ax_bar = plt.subplots()
ax_bar.bar(['category1', 'category2', 'category3'], np.random.randint(30, size=3))
plt.show()
