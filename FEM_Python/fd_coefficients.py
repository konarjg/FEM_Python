import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from math import factorial as fact

def fd_coefficients(m, n):
    p = int(np.floor((m + 1)/2) + (n - 2)/2)
    N = int(2 * p + 1)

    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(0, N):
        j = 0

        for k in range(-p, p + 1):
            A[i, j] = k**i
            j = j + 1

        if i == m:
            b[i] = fact(m)

    return sp.linalg.solve(A, b), p

def D0(a, x):
    N = len(x)
    A = np.zeros((N, N))

def D(a, x, m, n):
    N = len(x)
    A = np.zeros((N, N))
    d, p = fd_coefficients(m, n)
    dx = x[1] - x[0]

    for i in range(0, N):
        j = 0

        for k in range(-p, p):
            if 0 <= i + k < N:
                A[i + k, i] = A[i, i + k] = a(x[i]) * d[j]

            j = j + 1

    return A/dx**m

a2 = lambda x: -0.5
a0 = lambda x: -1/(x + 1e-9)

x = np.linspace(0, 5, 500)
H = D(a2, x, 6, 2) + 

E, u = sp.linalg.eigh(H)

plt.figure()
plt.plot(x, u[:, 0]**2)
plt.show()