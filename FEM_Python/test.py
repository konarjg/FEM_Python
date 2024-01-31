import math
import scipy as sp
import numpy as np

def cart(R):
    r, phi, theta = R
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return (x, y, z)

def dist(r1, r2, t2):
    return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * t2)

def grid():
   return [0, -np.sqrt(5 - 2 * np.sqrt(10/7))/3, np.sqrt(5 - 2 * np.sqrt(10/7))/3, -np.sqrt(5 + 2 * np.sqrt(10/7))/3, np.sqrt(5 + 2 * np.sqrt(10/7))/3] 

def grid4d():
    return [0, -np.sqrt(0.6), np.sqrt(0.6)]

def weights():
    return [128/225, (322 + 13 * np.sqrt(70))/900, (322 + 13 * np.sqrt(70))/900, (322 - 13 * np.sqrt(70))/900, (322 - 13 * np.sqrt(70))/900]

def weights4d():
    return [8/9, 5/9, 5/9]

def r(p):
    return -np.log(p)

def u(zeta, n):
    N = (2 * zeta)**n * np.sqrt(2 * zeta / math.factorial(2 * n))
    return lambda r: N * r**(n - 1) * np.exp(-zeta * r)

def D2(zeta, n):
    N = (2 * zeta)**n * np.sqrt(2 * zeta / math.factorial(2 * n))
    return lambda r: N * r**(n - 1) * np.exp(-zeta * r) * (zeta**2 - 2 * zeta/r)

def T_int(u1, D2):
    p = np.add(np.multiply(grid(), 0.5), 0.5)
    w = weights()
    T = 0

    for i in range(0, 5):
        T = T + 0.5 * u1(r(p[i])) * D2(r(p[i])) * r(p[i])**2 * 1/p[i] * w[i]

    return -2 * np.pi * T

def V_int(u1, u2, Z):
    p = np.add(np.multiply(grid(), 0.5), 0.5)
    w = weights()
    V = 0

    for i in range(0, 5):
        V = V + 0.5 * u1(r(p[i])) * u2(r(p[i])) * r(p[i]) * 1/p[i] * w[i]

    return -4 * Z * np.pi * V

def J_int(u1, u2, u3, u4):
    g = grid()
    p1 = p2 = np.add(np.multiply(g, 0.5), 0.5)
    t1 = t2 = g
    w = weights()

    J = 0

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                for l in range(0, 3):
                    r1 = r(p1[i])
                    r2 = r(p2[j])
                    dP = 1/p1[i] * 1/p2[j]
                    dR = r1**2 * r2**2
                    W = w[i] * w[j] * w[k] * w[l]

                    f = u1(r1) * u2(r1) * u3(r2) * u4(r2)
                    V = 1/dist(r1, r2, t2[l])

                    J = J + f * V * W * dR * dP

    return 4 * np.pi**2 * 0.5**2 * J

def S_int(u1, u2):
    p = np.add(np.multiply(grid(), 0.5), 0.5)
    w = weights()
    S = 0

    for i in range(0, 5):
        S = S + 0.5 * u1(r(p[i])) * u2(r(p[i])) * r(p[i])**2 * 1/p[i] * w[i]

    return 4 * np.pi * S

def T(psi, D2psi):
    n = len(psi)
    T = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            T[i, j] = T_int(psi[i], D2psi[j])

    return T

def V(psi, Z):
    n = len(psi)
    V = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            V[i, j] = V_int(psi[i], psi[j], Z)

    return V

def G(P1, psi1):
    n1 = len(psi1)
    G = np.zeros((n1, n1))

    for i in range(0, n1):
        for j in range(0, n1):
            for k in range(0, n1):
                for l in range(0, n1):
                    G[i, j] = G[i, j] + P1[k, l] * (J_int(psi1[i], psi1[j], psi1[k], psi1[l]) - J_int(psi1[i], psi1[l], psi1[j], psi1[k]))
    
    return G

def S(psi):
    n = len(psi)
    S = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            S[i, j] = S_int(psi[i], psi[j])

    return S

def P(C, n):
    P = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                P[i, j] = P[i, j] + C[i, k] * C[j, k]
                
    return P

def helium(Z):
    a, b = Z[0], Z[1]

    psi1 = [u(a, 1)]
    D2psi1 = [D2(a, 1)]

    psi2 = [u(b, 1)]
    D2psi2 = [D2(b, 1)]

    n1 = len(psi1)
    n2 = len(psi2)

    H1 = T(psi1, D2psi1) + V(psi1, n1 + n2)
    S1 = S(psi1)

    H2 = T(psi2, D2psi2) + V(psi2, n1 + n2)
    S2 = S(psi2)

    epsilon1, C1 = sp.linalg.eigh(H1, S1)
    epsilon2, C2 = sp.linalg.eigh(H2, S2)

    P1 = P(C1, n1)
    P2 = P(C2, n2)
    E0 = epsilon1.sum() + epsilon2.sum() + (P1 * H1).sum() + (P2 * H2).sum()

    for i in range(0, 1000):
        G1 = G(P1, psi1)
        G2 = G(P2, psi2)
        F1 = H1 + G1
        F2 = H2 + G2

        epsilon1, C1 = sp.linalg.eigh(F1, S1)
        epsilon2, C2 = sp.linalg.eigh(F2, S2)

        Pi1 = P(C1, n1)
        Pi2 = P(C2, n2)
        Ei = epsilon1.sum() + epsilon2.sum() + (P1 * H1).sum() + (P2 * H2).sum()

        if np.abs(Ei - E0) < 1e-6:
            E0 = Ei
            break

        P1 = Pi1
        P2 = Pi2
        E0 = Ei

    print((a, b, E0))
    return E0

def lithium(Z):
    a, b, c = Z[0], Z[1], Z[2]

    psi1 = [u(a, 1), u(c, 2)]
    D2psi1 = [D2(a, 1), D2(c, 2)]

    psi2 = [u(b, 1)]
    D2psi2 = [D2(b, 1)]

    n1 = len(psi1)
    n2 = len(psi2)

    H1 = T(psi1, D2psi1) + V(psi1, n1 + n2)
    S1 = S(psi1)

    H2 = T(psi2, D2psi2) + V(psi2, n1 + n2)
    S2 = S(psi2)

    epsilon1, C1 = sp.linalg.eigh(H1, S1)
    epsilon2, C2 = sp.linalg.eigh(H2, S2)

    P1 = P(C1, n1)
    P2 = P(C2, n2)
    E0 = epsilon1.sum() + epsilon2.sum()

    for i in range(0, 1000):
        F1 = H1 + G(P1, P2, psi1, psi2)
        F2 = H2 + G(P2, P1, psi2, psi1)

        epsilon1, C1 = sp.linalg.eigh(F1, S1)
        epsilon2, C2 = sp.linalg.eigh(F2, S2)

        Pi1 = P(C1, n1)
        Pi2 = P(C2, n2)
        Ei = epsilon1.sum() + epsilon2.sum()

        if np.abs(Ei - E0) < 1e-6:
            E0 = Ei
            break

        P1 = Pi1
        P2 = Pi2
        E0 = Ei

    print((a, b, E0))
    return E0

result_he = sp.optimize.minimize(helium, [1.7, 2])
#result_li = sp.optimize.minimize(lithium, [1.5, 2.15, 3])
print(result_he)