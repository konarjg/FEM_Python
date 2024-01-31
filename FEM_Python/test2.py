import scipy as sp
import numpy as np
import sympy as sm
from sympy.printing import pprint

def cart(r, phi, theta):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return (x, y, z)

def dist(r1, phi1, theta1, r2, phi2, theta2):
    x1, y1, z1 = cart(r1, phi1, theta1)
    x2, y2, z2 = cart(r2, phi2, theta2)

    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def STO(a, r):
    return np.exp(-a * r)

def D2_STO(a, r):
     return STO(a, r) * (a**2 - 2 * a/r)

def u(a, b, r1, r2):
    return STO(a, r1) * STO(b, r2)

def T_int(a, b, r1, r2):
    T_u = -0.5 * STO(b, r2) * D2_STO(a, r1) - 0.5 * STO(a, r1) * D2_STO(b, r2)
    return u(a, b, r1, r2) * T_u * r1**2 * r2**2 * 16 * np.pi**2

def V_int(a, b, r1, r2, Z):
    V = -Z/r1 - Z/r2
    return u(a, b, r1, r2)**2 * V * r1**2 * r2**2 * 16 * np.pi**2

def H(a, b):
    return sp.integrate.dblquad(lambda r1, r2: T_int(a, b, r1, r2) + V_int(a, b, r1, r2, 2), 0, np.inf, 0, np.inf)[0]

def J_int(a, b, R):
    if len(np.shape(R)) < 2:
        R = np.reshape(R, (len(R), 1))

    (d, n) = np.shape(R)
    A = np.zeros(n)

    for i in range(0, n):
        t1, phi1, theta1, t2, phi2, theta2 = R[0, i], R[1, i], R[2, i], R[3, i], R[4, i], R[5, i]
        r1, r2 = -np.log(t1), -np.log(t2)
        r12 = dist(r1, phi1, theta1, r2, phi2, theta2)

        A[i] = 1/t1 * 1/t2 * u(a, b, r1, r2) **2 * 1/(r12 + 1e-6) * r1**2 * r2**2 * np.sin(theta1) * np.sin(theta2)

    return A

def J(a, b):
    A = np.zeros(6)
    B = [1, 2 * np.pi, np.pi, 1, 2 * np.pi, np.pi]
    rng = sp.stats.qmc.Sobol(6)

    return sp.integrate.qmc_quad(lambda R: J_int(a, b, R), A, B, qrng=rng, n_points=2**16, n_estimates=1)[0]

def S(a, b):
    return sp.integrate.dblquad(lambda r1, r2: u(a, b, r1, r2)**2 * r1**2 * r2**2 * 16 * np.pi**2, 0, np.inf, 0, np.inf)[0]

def E(alpha):
    a = alpha[0]
    E = (H(a, a) + J(a, a))/S(a, a)
    print((a, E))
    return E

result = sp.optimize.minimize(E, [1.7])
