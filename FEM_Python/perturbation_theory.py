import scipy as sp
import numpy as np

V_func = lambda r1, r2, r3: 1/np.abs(r2 - r1 + 0.001) + 1/np.abs(r3 - r2 + 0.001) + 1/np.abs(r3 - r1 + 0.001)
u_func = lambda r1, r2, r3: np.exp(-2*(r1 + r2 + r3))

R1 = np.linspace(0.001, 200, 20)
R2 = np.linspace(0.001, 200, 20)
R3 = np.linspace(0.001, 200, 20)

V = np.matrix((20**3, 20**3))
u_T = []

for i in range(0, 20):
    for j in range(0, 20):
        for k in range(0, 20):
            u_T[400 * i + 20 * j + k] = u_func(R1[i], R2[j], R3[k])
            V[400 * i + 20 * j + k, 400 * i + 20 * j + k] = V_func(R1[i], R2[j], R3[k])

u_T = np.atleast_2d(u_T)
u = np.tranpose(u_T)

print(-30.6 + )