import scipy as sp
import numpy as np
import math
from matplotlib import pyplot as plt
import test as op

def cart(R):
    (r, phi, theta) = R
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return (x, y, z)

def dist(R1, R2):
    (x1, y1, z1) = cart(R1)
    (x2, y2, z2) = cart(R2)

    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def STO(n, zeta, r):
    N = (2 * zeta)**n * np.sqrt(2 * zeta / math.factorial(2 * n))
    f = r ** (n - 1) * np.exp(-zeta * r)

    return N * f

def D2_STO(n, zeta, r):
    a = n * (n - 1) * np.float_power(r, -2)
    b = -2 * n * zeta * np.float_power(r, -1)
    c = zeta**2

    return np.add(a + b, c) * STO(n, zeta, r)

def x(t):
    return -np.log(t)

def H_integral(psi_r, psi_s, D2s, Z, R):
    if len(np.shape(R)) < 2:
        R = np.reshape(R, (len(R), 1))

    t = R[0, :]
    r = x(t)
    T = -0.5 * D2s(r)
    V = -Z * np.float_power(r, -1) * psi_s(r)

    return psi_r(r) * (T + V) * 4 * np.pi * r**2 * np.float_power(t, -1)

def H_core(psi, D2, N, Z):
    H = np.zeros((N, N))
    a = [0]
    b = [1]

    for r in range(0, N):
        for s in range(0, N):
            H[r, s] = sp.integrate.qmc_quad(lambda R: H_integral(psi[r], psi[s], D2[s], Z, R), a, b, n_estimates=40)[0]

    return H

def J_integral(psi_r, psi_s, psi_t, psi_u, R):
    if len(np.shape(R)) < 2:
        R = np.reshape(R, (len(R), 1))

    (t1, t2) = (R[0, :], R[3, :])
    R1 = (r1, phi1, theta1) = (x(t1), R[1, :], R[2, :])
    R2 = (r2, phi2, theta2) = (x(t2), R[4, :], R[5, :])

    r12 = dist(R1, R2)

    return psi_r(r1) * psi_s(r1) * np.float_power(r12, -1) * psi_t(r2) * psi_u(r2) * r1**2 * r2**2 * np.sin(theta1) * np.sin(theta2) * np.float_power(t1, -1) * np.float_power(t2, -1)

def J(psi, N, Z):
    J = np.zeros((N, N, N, N))
    a = np.zeros(6)
    b = [1, 2 * np.pi, np.pi, 1, 2 * np.pi, np.pi]

    for r in range(0, N):
        for s in range(0, N):
            for t in range(0, N):
                for u in range(0, N):
                    J[r, s, t, u] = sp.integrate.qmc_quad(lambda R: J_integral(psi[r], psi[s], psi[t], psi[u], R), a, b, n_estimates=40)[0]

    return J

def P(C, N):
    P = np.zeros((C.shape[0], C.shape[0]))

    for t in range(C.shape[0]):
        for u in range(C.shape[0]):
            for j in range(int(N/2)):
                P[t, u] += 2 * C[t, j] * C[u, j]

    return P

def G(P, J, N):
    G = np.zeros((N, N))

    for r in range(0, N):
        for s in range(0, N):
            g = 0

            for t in range(0, N):
                for u in range(0, N):
                    int1 = J[r, s, t, u]
                    int2 = J[r, u, t, s]

                    g = g + P[t, u] * (int1 - 0.5 * int2)

            G[r, s] = g

    return G

def S_integral(psi_r, psi_s, R):
    if len(np.shape(R)) < 2:
        R = np.reshape(R, (len(R), 1))

    t = R[0, :]
    r = x(t)

    return psi_r(r) * psi_s(r) * 4 * np.pi * r**2 * np.float_power(t, -1)

def S(psi, N):
    S = np.zeros((N, N))
    a = [0]
    b = [1]

    for r in range(0, N):
        for s in range(0, N):
            S[r, s] = sp.integrate.qmc_quad(lambda R: S_integral(psi[r], psi[s], R), a, b, n_estimates=40)[0]

    return S

def F(psi, H, P, N, Z):
    J0 = J(psi, N, Z)
    G0 = G(P, J0, N)

    return H + G0

def E(epsilon, N, P, H):
    E = 0

    for i in range(0, int(N/2)):
        E = E + epsilon[i].real

    return E + 0.5 * (P * H).sum()

def SCF(zeta, max_iterations=1000):
    (a, b) = (zeta[0], zeta[1])

    psi1 = lambda r: STO(1, a, r)
    psi2 = lambda r: STO(1, b, r)
    D21 = lambda r: D2_STO(1, a, r)
    D22 = lambda r: D2_STO(1, b, r)

    psi = [psi1, psi2]
    D2 = [D21, D22]
    N = len(psi)
    NE = Z

    H = H_core(psi, D2, N, Z)

    print(H)
    S0 = S(psi, N)
    (epsilon, C) = sp.linalg.eigh(H, S0)

    P0 = P(C, NE)

    threshold = 0.001
    prev_E0 = E(epsilon, NE, P0, H)

    for i in range(0, max_iterations):
        F0 = F(psi, H, P0, N, Z)

        (epsilon, C) = sp.linalg.eigh(F0, S0)

        E0 = E(epsilon, NE, P0, H)

        if np.abs(prev_E0 - E0) < threshold:
            prev_E0 = E0
            break

        P0 = P(C, NE)
        prev_E0 = E0

    return prev_E0

Z = 2
print(SCF([1.7, 2]))

