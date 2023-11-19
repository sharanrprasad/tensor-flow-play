from sympy import symbols, diff

J, w = symbols('J, w')

# For this testing, let's assume that the the cost function J is equal to w2
J = w**2

dJ_dw = diff(J,w)

print(dJ_dw.subs([(w,2)]))




