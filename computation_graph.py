# We know that gradient descent formula (w here is a model paramaeter) w = w - learning_rate(dJ/dw) Let's assume that
# J = (2 + 3w)^2 for this exercise And a = 2 + 3w so that makes J = a^2 dJ/da = 2a Next step is compute dJ/dw but
# before that let's calculate da/dw = d(2+3w)/dw = 3 now dJ/dw can be written as = dJ/da * da/dw ( chain rule of
# calculus. This is what reduceses the complexity from N * P to N + p) and makes machine learning efficient dJ/dw =
# 2(2+3w) * 3 = 6(2+3w)

# This keeps going back in the neural network until we reach layer 1 values.


# Below is the representation of the same using sumpy

from sympy import symbols, diff

w = 3
a = 2 + 3 * w
J = a ** 2
print(f"a = {a}, J = {J}")

sw, sJ, sa = symbols('w,J,a')
sJ = sa ** 2
print(sJ)

dJ_da = diff(sJ, sa)
print(dJ_da)

sa = 2 + 3 * sw

da_dw = diff(sa, sw)
print(da_dw)

dJ_dw = da_dw * dJ_da
print(dJ_dw)

# On a side note about slopes -
# If the slope value is positive then as W goes up J also goes up
# If slope value is negative then as W goes up J goes down. (Calculate using y = mx+b)
# Our aim is to bring J to zero (Cost function) so we need to try and reduce W(W = W - dJ/dW) but if the
# slope is negative it will go up and still reduce J value.
