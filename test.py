# import cvxpy as cp x  =  cp.Variable(1)
# objective = 16 * x[0]**(1/3) * x[1]**(2/3)
# objective =cp.Maximize(objective)
# constraints = [0.0000001 <=x,  sum(x) == 1 ]
# prob = cp.Problem(objective, constraints)
# result = prob.solve()
#
# y  =  cp.Variable(2)
import math
import pdb

eps = 1e-6
b = 1 + eps
a = 1 - eps
v3 = (
    math.log((b - a) / (math.log(b) - math.log(a)))
    - math.log(a * b)
    + (b * math.log(b) - a * math.log(a)) / (b - a)
    - 1
)
m = 100
rho = 3 * m ** (1 / 3) * (eps)**(-5/3)
x = math.log((1 + eps) / (1 - eps)) / (2 * eps)
v1 = x - math.log(x) - 1 - 1/2*math.log((1 + eps) * (1 - eps))
v2 = math.log(1 + eps)
print("v1 v2", v1, v2)


case1 = (
    math.log(m)
    * (2 * eps ** 2 / rho - math.log(1 - eps))
    / (eps ** 3 / (2 * rho ** 2) - math.log(1 - eps) ** 2)
)

case2 = (
    math.log(m)
    * (eps / (2 * rho) - 2 * math.log(1 - eps))
    / (eps ** 3 / (2 * rho ** 2) - math.log(1 - eps) ** 2)
)

case3 = eps**2 / (2*rho) + math.log(1 - eps)


case4 = (1 + eps)/math.exp(eps/rho)
print(case4)

# print(case1, case2, case3)
# objective2 = y[0]*y[1] /(y[0]*1 +  y[0]*4)
# objective2 =cp.Maximize(objective2)
# constraints2= [0.0000001 <=y,  sum(y) == 1 ]
# prob1 = cp.Problem(objective2, constraints2)
# result1 = prob1.solve()
# print("The second", y.value)


# matplotlib.use('Agg')
# import numpy as np
# from matplotlib import pyplot as plt
# c1 = 1
# c2 = 4
# x = np.array(range(0,100,1))/100
# y = x**(c1/(c1+c2))*(1-x)**(c2/(c1+c2))

# y2 = x*(1-x)/(x*c2**2+ (1-x)*c1**2)
# plt.plot(x,y)
# plt.plot(x,y2)
# a= np.argmax(y)
# print(a)
# print(np.argmax(y2))
# print(x[a])

# plt.show()

# plt.savefig('f1.png')

