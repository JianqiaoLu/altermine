import math
from graph_tool.flow import min_cut
from graph_tool.topology import random_spanning_tree
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import cvxpy as cp

from scipy.sparse.linalg import spsolve

# # Input Format
#
# Consider an undirected graph G=(V,E). Let n=|V| and m=|E| denote the numbers of vertices and edges respectively. Assume WLOG that V={0,1,2,…,n−1} with s=0 and t=n−1. The input specifies n and a list of edges each of which is given by a tuple (i,j,c), i.e., the vertices i and j and the cap c.

# In[ ]:
global iter
ter = 0
global want
want = 1

global eps
eps = 1e-2


def test_graph_1():
    return 6, [
        [0, 1, 20],
        [1, 2, 10],
        [1, 3, 10],
        [0, 2, 4],
        [2, 3, 9],
        [2, 4, 14],
        [3, 5, 20],
        [3, 4, 7],
        [4, 5, 4],
    ]


def test_graph_2():
    return 7, [
        [0, 1, 220],
        [0, 2, 54],
        [1, 2, 510],
        [1, 3, 910],
        [2, 3, 90],
        [2, 4, 114],
        [3, 4, 37],
        [3, 5, 120],
        [4, 5, 54],
        [3, 6, 120],
        [2, 6, 310],
        [1, 6, 151],
        [4, 6, 103],
    ]


def test_graph_3():
    return 8, [
        [0, 1, 20],
        [0, 2, 12],
        [0, 3, 19],
        [1, 4, 12],
        [2, 5, 6],
        [3, 6, 8],
        [4, 7, 12],
        [5, 7, 12],
        [6, 7, 11],
    ]


def test_graph_4():
    return 6, [[0, 1, 20], [1, 2, 8], [2, 3, 6], [2, 4, 14], [3, 4, 6], [4, 5, 12]]


def test_graph_5():
    return 14, [
        [0, 1, 20],
        [0, 2, 8],
        [1, 3, 6],
        [1, 4, 19],
        [2, 5, 14],
        [2, 6, 9],
        [3, 7, 13],
        [3, 8, 21],
        [4, 9, 29],
        [4, 10, 12],
        [5, 11, 31],
        [6, 12, 17],
        [7, 13, 5],
        [8, 13, 17],
        [9, 13, 28],
        [10, 13, 10],
        [11, 13, 11],
        [12, 13, 7],
    ]


def test_graph_6():

    return 6, [
        [0, 1, 20],
        [0, 2, 24],
        [1, 3, 19],
        [2, 3, 15],
        [4, 3, 11],
        [2, 4, 14],
        [1, 4, 37],
        [3, 5, 48],
    ]


def test_graph_7():
    return 4, [[0, 1, 2], [1, 2, 1], [2, 3, 2]]


def test_graph_8():
    return 3, [
        [0, 1, math.sqrt(5)],
        [1, 2, 2],
    ]


def test_graph_9():
    return 3, [
        [0, 1, 5],
        [1, 2, 10],
        [0, 2, 1],
    ]


def test_graph_10():
    return 6, [
        [0, 3, 1.5],
        [0, 4, 1],
        [3, 1, 2],
        [4, 1, 2],
        [0, 1, 1.5],
        [1, 2, 10],
        [2, 5, 0.5],
        [3, 5, 1],
        [1, 5, 2.5],
    ]


def test_graph_cong():
    return 10, [
        [0, 9, 1],
        [0, 1, 1],
        [0, 2, 1],
        [0, 3, 1],
        [0, 4, 1],
        [1, 5, 1],
        [2, 6, 1],
        [3, 7, 1],
        [4, 8, 1],
        [5, 9, 1],
        [6, 9, 1],
        [7, 9, 1],
        [8, 9, 1],
    ]


def test_graph(k):
    n = k * (k - 1) + 2
    vertex_list = np.array([i for i in range(1, k + 1)])
    lists = []

    for i in range(k - 1):
        lists.append(vertex_list + i * k)

    edges = [[0, i, 1] for i in lists[0]]

    for j in range(k - 2):
        for i in range(k):
            edges.append([lists[j][i], lists[j + 1][i], 1])

    for i in range(k):
        edges.append([lists[-1][i], k * (k - 1) + 1, 1])

    edges.append([0, k * (k - 1) + 1, 1])

    return n, edges


# In[ ]:


def test_graph_unit_cap():
    return 6, [
        [0, 1, 1],
        [0, 2, 1],
        [1, 2, 1],
        [1, 3, 1],
        #    [2, 3, 1],
        [2, 4, 1],
        [3, 4, 1],
        [3, 5, 1],
        [4, 5, 1],
    ]


# def test_graph_1():
#     return 6, [
#         [0, 1, 20], [1, 2, 10],
#                [1, 3, 10],
#         [0, 2, 4],
#         [2, 3, 9],
#         [2, 4, 14],
#         [3, 5, 20],
#         [3, 4, 7],
#         [4, 5, 4],
#     ]


# def test_graph_2():
#     return 7, [
#         [0, 1, 220],
#         [0, 2, 54],
#         [1, 2, 510],
#         [1, 3, 910],
#         [2, 3, 90],
#         [2, 4, 114],
#         [3, 4, 37],
#         [3, 5, 120],
#         [4, 5, 54],
#         [3, 6, 120],
#         [2, 6, 310],
#         [1, 6, 151],
#         [4, 6, 103],
#     ]


# def test_graph_3():
#     return 8, [
#         [0, 1, 20],
#         [0, 2, 12],
#         [0, 3, 19],
#         [1, 4, 12],
#         [2, 5, 6],
#         [3, 6, 8],
#         [4, 7, 12],
#         [5, 7, 12],
#         [6, 7, 11],
#     ]


# def test_graph_4():
#     return 6, [[0, 1, 20], [1, 2, 8], [2, 3, 6], [2, 4, 14], [3, 4, 6], [4, 5, 12]]


# def test_graph_5():
#     return 14, [
#         [0, 1, 20],
#         [0, 2, 8],
#         [1, 3, 6],
#         [1, 4, 19],
#         [2, 5, 14],
#         [2, 6, 9],
#         [3, 7, 13],
#         [3, 8, 21],
#         [4, 9, 29],
#         [4, 10, 12],
#         [5, 11, 31],
#         [6, 12, 17],
#         [7, 13, 5],
#         [8, 13, 17],
#         [9, 13, 28],
#         [10, 13, 10],
#         [11, 13, 11],
#         [12, 13, 7],
#     ]


# def test_graph_6():

#     return 6, [
#         [0, 1, 20],
#         [0, 2, 24],
#         [1, 3, 19],
#         [2, 3, 15],
#         [4, 3, 11],
#         [2, 4, 14],
#         [1, 4, 37],
#         [3, 5, 48],
#     ]

# def test_graph_7():
#     return 3, [
#       [0, 1, ],
#       [1, 1, ],
#     [1, 2, ],
#  ]


# def test_graph_9():
#     return 4, [
#         [0, 1, 2],
#       [0, 1, ],[1, 2, 10],
#         [2,3, ],
# ]


# def test_graph_unit_cap():
#     return 6, [
#         [0, 1, 1],
#         [0, 2, 1],
#         [1, 2, 1],
#         [1, 3, 1],
#         #    [2, 3, 1],
#         [2, 4, 1],
#         [3, 4, 1],
#         [3, 5, 1],
#         [4, 5, 1],
#     ]


# In[ ]:


# n, edge = test_graph_6()
# n, edge = test_graph_unit_cap()
# min_cuts = [0,1] for test_graph_1
# min_cuts = [0,1] for test_graph_2
# min_cuts  = [3,4,5] for test_graph_3
# min_cuts = [1] for test_graph_4
# min_cuts = [0,1] for test_graph_5


# print('1111') # print(n) # In[ ]:


def electrical_flow(n, res):
    # res is expressed in its inverse
    A = np.zeros([n, n])
    for i, j, r in res:
        A[i, j] -= r
        A[j, i] -= r
        A[i, i] += r
        A[j, j] += r

    A[0, :] = np.zeros(n)
    A[0, 0] = 1.0
    A[n - 1, :] = np.zeros(n)
    A[n - 1, n - 1] = 1.0

    b = np.zeros(n)
    b[0] = 1
    b[n - 1] = 0.0
    # add 1-0 cosntraint to   variable phi
    # try:
    #  phi = np.linalg.inv(A) @ b
    # except:
    phi = spsolve(A, b)
    # phi = np.linalg.inv(A) @ b

    # @ operation just work as np.dot
    flow = [[i, j, (phi[i] - phi[j]) * r] for i, j, r in res]
    energy = sum([(phi[i] - phi[j]) ** 2 * r for i, j, r in res])

    return phi, flow, energy


# In[ ]:


# phi, flow, energy = electrical_flow(n, edge)
# print(phi)
# print(flow)
# print("energy:",energy)


# In[ ]:


def update_cvx(phi, edge):
    global epsilon
    x = cp.Variable(len(edge))
    I = np.ones(len(edge))
    objective = 0

    for k in range(len(edge)):
        phi1 = phi[edge[k][0]]
        phi2 = phi[edge[k][1]]
        cap = edge[k][2]
        objective += ((phi1 - phi2) ** 2 * cap ** 2) * cp.inv_pos(x[k])
    # 原来倒数需要用inv_pos
    objective = cp.Minimize(objective)
    constraints = [0.000001 <= x, sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    # please remember that we can speficy using one index of some variable
    return x.value


def update_w(phi, edge):
    eps_m = 0.0001 / len(edge)
    W = sum([abs(phi[i] - phi[j]) * c for i, j, c in edge])
    w = [1 / W * abs(phi[i] - phi[j]) * c for i, j, c in edge]
    w_hat = [i for i in range(len(w)) if w[i] < eps_m]
    pre = []

    while len(w_hat):
        if pre == w_hat:
            break

        for k in w_hat:
            W -= abs(phi[edge[k][0]] - phi[edge[k][1]]) * edge[k][2]

        w = [
            (1 - len(w_hat) * eps_m) / W * abs(phi[i] - phi[j]) * c for i, j, c in edge
        ]
        for k in w_hat:
            w[k] = eps_m
        # if iter % 10 == 0:
        #   print(iter, w_hat)
        pre = w_hat
        w_hat = [i for i in range(0, len(w)) if w[i] <= eps_m]
    # if iter % 10 == 0:
    #   import pdb
    #   pdb.set_trace()

    return w

    # while min(uc_wets) < eps_m:
    #   index  = uc_wets.index(min(uc_wets))
    #   uc_wets[index]  = eps_m

    # return


# In[ ]:


def calnu(w, min_cuts, cap_comp, data4, data5):
    flag1 = "y"
    for j in range(len(min_cuts)):
        nu = 1
        eta = 0
        for i in min_cuts[j]:
            nu *= w[i] ** (cap_comp[i])
            eta += w[i]
        if data5[j] != [] and flag1 == "y":
            if nu < data4[j][-1]:
                flag1 = "n"
        data4[j].append(nu)
        data5[j].append(eta)
    return data4, data5


def caljensen(min_cuts, cap_comp, w_comp, data6):
    id = 1
    for min_cut in min_cuts:
        lhs = sum([cap_comp[i] * math.log(w_comp[i]) for i in min_cut])
        rhs = math.log(sum([cap_comp[i] * w_comp[i] for i in min_cut]))
        data6[id].append(rhs - lhs)
        id += 1


def update_data(phi, energy, w, data1, data2, data3):
    data1.append(phi)
    data2.append(energy)
    data3.append(w)


def calab(min_cuts, w_comp, data7):
    id = 0
    for min_cut in min_cuts:
        w_range = [w_comp[i] for i in min_cut]
        data7[id].append([min(w_range), max(w_range)])
        id += 1


def calflag(min_cuts, data6):
    flag = "y"
    for i in range(1, len(min_cuts) + 1):
        if data6[0][-1] < data6[i][-1]:
            flag = "n"
    return flag


def getpedge(i):
    eps = 1e-2
    m = i - 1
    edge = [[0, 1, 1]]
    w0 = [eps / m] + [(1 - eps / m) / (m - 1)] * (m - 1)
    for x in range(1, 100):
        x = 2
        for j in range(1, i - 1):
            edge.append([j, j + 1, x])

        res = [[edge[i][0], edge[i][1], edge[i][2] ** 2 / w0[i]]
               for i in range(m)]
        phi, flow, energy_phi = electrical_flow(i, res)
        cap = [x[2] for x in edge]
        cong = [abs(flow[i][2]) / cap[i] for i in range(m)]
        gamma_w = [cong[i] ** 2 / energy_phi for i in range(m)]
        gamma = [w0[i] * cong[i] ** 2 / energy_phi for i in range(m)]
        ans = [w0[i] * (math.sqrt(gamma_w[i]) - 1) ** 2 for i in range(m)]

        if abs(gamma_w[-1] - 1) < 1e-2:
            import pdb

            pdb.set_trace()

            return i, edge


def nparrelledge(i):

    global eps
    m = i - 1
    edge = [[0, 1, 1]]
    w0 = [eps / m] + [(1 - eps / m) / (m - 1)] * (m - 1)
    x = 1
    for j in range(1, i - 1):
        edge.append([j, j + 1, x])
    res = [[edge[i][0], edge[i][1], edge[i][2] ** 2 / w0[i]] for i in range(m)]
    phi, flow, energy_phi = electrical_flow(i, res)
    cap = [x[2] for x in edge]
    cong = [abs(flow[i][2]) / cap[i] for i in range(m)]
    gamma_w = [cong[i] ** 2 / energy_phi for i in range(m)]
    gamma = [w0[i] * cong[i] ** 2 / energy_phi for i in range(m)]
    ans = [w0[i] * (math.sqrt(gamma_w[i]) - 1) ** 2 for i in range(m)]
    # if abs(gamma_w[-1] - 1) < 1e-2:
    # import pdb
    # pdb.set_trace()

    return i, edge, w0, res, ans, x


def onestepmini(i):

    data = [[], []]
    n, edge, w0, res, reduce, othercap = nparrelledge(i)
    m = len(edge)
    phi, flow, energy_phi_0 = electrical_flow(n, res)
    data[0].append(energy_phi_0)
    data[1].append(math.exp(-sum(reduce)))

    w1 = update_w(phi, edge)
    res = [[edge[i][0], edge[i][1], edge[i][2] ** 2 / w1[i]] for i in range(m)]
    phi, flow, energy_phi_1 = electrical_flow(n, res)

    data[0].append(energy_phi_1)
    data[1].append(energy_phi_1 / energy_phi_0)
    return data


def altertating_minimization_simple(n, edge, min_cuts=[], cut_val=1):

    global eps
    m = len(edge)
    data2 = [[]]
    w0 = [eps/m,0.9-eps/m,0.1]
    # w0 = [eps / m] + [(1 - eps / m) / (m - 1)] * (m - 1)
    w = w0
    res = [[edge[i][0], edge[i][1], edge[i][2] ** 2 / w0[i]] for i in range(m)]

    for round in range(1000):
        phi, flow, energy_phi = electrical_flow(n, res)
        import pdb 
        pdb.set_trace()
        

        ans2 = sum([(phi[i] - phi[j]) ** 2 * r for i, j, r in res])
        # res = [[i, j, m* c ** 2] for i, j, c in edge]
        r_e = [1 / r for i, j, r in res]
        R = sum(r_e)
        ans = 1 / R

        energy_e = [(phi[i] - phi[j]) ** 2 * r for i, j, r in res]
        gamma = [i / sum(energy_e) for i in energy_e]
        gamma_w = [gamma[i] / w[i] for i in range(m)]

        if len(data2) and abs(data2[-1] - (energy_phi)) < 1e-5:
            break
        data2.append(energy_phi)
        w = update_w(phi, edge)
        res = [[edge[i][0], edge[i][1], edge[i][2] ** 2 / w[i]]
               for i in range(m)]
        energy_w = sum([(phi[i] - phi[j]) ** 2 * r for i, j, r in res])

    return phi, math.sqrt(energy_phi), data2, "-"


def altertating_minimization(n, edge):

    global eps
    m = len(edge)
    # eps = .01/m

    # w0 = np.random.dirichlet(np.ones(m), size=1)[0]
    w0 = [1 / m for i in range(m)]
    # w0 = [eps / m] + [(1 - eps / m) / (m - 1)] * (m - 1)
    # w0 = [ 0.49, 0.005, 0.5, 0.005]
    res = [[edge[i][0], edge[i][1], edge[i][2] ** 2 / w0[i]] for i in range(m)]
    pre_energy = 0
    data = []

    for round in range(1000):
        phi, flow, energy = electrical_flow(n, res)

        data.append(energy)

        if abs(energy - pre_energy) < 1e-5:
            break

        w = update_w(phi, edge)

        res = [[edge[i][0], edge[i][1], edge[i][2] ** 2 / w[i]]
               for i in range(m)]

        pre_energy = energy

    return data


# In[ ]: j
# phi, flow , data, data2,data3, data4= altertating_minimization(n, edge)
# df = pd.DataFrame(data = data)
# df2 = pd.DataFrame(data = data2)
# df3 = pd.DataFrame(data = data3)
# df4 = pd.DataFrame(data = data4)
# df2.to_excel("energy.xlsx")
# df3.to_excel("wj.xlsx")

# df4.to_excel("nu.xlsx")
#
# w_1 = data
# x = [i for i in range(0,len(w_1))]
# w_1 = list(w_1)
#
#
# plt.plot(w_1)
# plt.show()
