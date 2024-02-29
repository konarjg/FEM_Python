import scipy as sp
import numpy as np
import random as rand
import matplotlib.pyplot as plt

def V(x, y, n):
    r = np.sqrt(x**2 + y**2)

    if r >= n + 1 or r == 0:
        return 1e12

    return -1/(r + 0.0001)

def H(n):
    x = np.linspace(-n - 1, n + 1, n)
    y = np.linspace(-n - 1, n + 1, n)

    m = n

    A = np.zeros((m**2, m**2))
    B = np.zeros((m**2, m**2))
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    for k in range(0, n):
        for j in range(0, n):
            i = m * k + j

            if i + m < m**2:
                A[i + m, i] = A[i, i + m] = 1/dx**2

            if i - m >= 0:
                A[i - m, i] = A[i, i - m] = 1/dx**2

            if i + 1 < m**2:
                A[i + 1, i] = A[i, i + 1] = 1/dy**2

            if i - 1 >= 0:
                A[i - 1, i] = A[i, i - 1] = 1/dy**2

            A[i, i] = -2/dx**2 - 2/dy**2
            B[i, i] = V(x[k], y[j], n + 2)

    return -0.5 * A + B, x, y

n = int(np.sqrt(2000))
H, x, y = H(n)
E, u = sp.linalg.eigh(H)

for i in range(0, 5):
    print(-13.6/(2 * i + 1)**2)

print(np.unique(E)[0:5] * 27.2)