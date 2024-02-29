import math
import scipy as sp
import numpy as np
import sympy as sm
from sympy.printing import pprint
from sympy import oo

def dist(A, B):
    Ax, Ay, Az = A[0], A[1], A[2]
    Bx, By, Bz = B[0], B[1], B[2]
    
    return np.sqrt((Ax - Bx)**2 + (Ay - By)**2 + (Az - Bz)**2)

def F0(x):
    f = lambda t: np.exp(-x * t**2)
    return sp.integrate.quad(f, 0, 1)[0]

def coulomb(A, B, C, D, Rab2, Rcd2, Rpq2):
    return 2.0*(np.pi**2.5)/((A+B)*(C+D)*np.sqrt(A+B+C+D))*F0((A+B)*(C+D)*Rpq2/(A+B+C+D))*np.exp(-A*B*Rab2/(A+B)-C*D*Rcd2/(C+D))

def overlap(A, B, Rab2):
    return (np.pi/(A + B))**1.5*np.exp(-A * B * Rab2 /(A + B))

def kinetic(A, B, Rab2):
    return A*B/(A+B)*(3.0-2.0*A*B*Rab2/(A+B))*(np.pi/(A+B))**1.5*np.exp(-A*B*Rab2/(A+B))

def nuclear(A,B,Rab2,Rcp2,Zc):
    V = 2.0*np.pi/(A+B)*F0((A+B)*Rcp2)*np.exp(-A*B*Rab2/(A+B))
    return -V*Zc

def S_GTO(a, b, A, B):
    Rab2 = dist(A, B)**2
    return overlap(a, b, Rab2)

def T_GTO(a, b, A, B):
    Rab2 = dist(A, B)**2
    return kinetic(a, b, Rab2)

def V_GTO(Z, a, b, A, B):
    s1 = 1/(2 * a + 2 * b)
    P = (2 * a * A + 2 * b * B) * s1
    Rab2 = dist(A, B)**2
    Rap2 = dist(A, P)**2

    return nuclear(a, b, Rab2, Rap2, Z)

def J_GTO(a, b, c, d, A, B, C, D):
    s1 = 1/(2 * a + 2 * b)
    s2 = 1/(2 * c + 2 * d)
    P = (2 * a * A + 2 * b * B) * s1
    Q = (2 * c * C + 2 * d * D) * s2

    Rab2 = dist(A, B)**2
    Rcd2 = dist(C, D)**2
    Rpq2 = dist(P, Q)**2

    return coulomb(a, b, c, d, Rab2, Rcd2, Rpq2)

def S_int(a1, d1, N1, a2, d2, N2, A):
    J = 0

    for i in range(0, 3):
        for j in range(0, 3):
            J = J + N1[i] * N2[j] * d1[i] * d2[j] * S_GTO(a1[i], a2[j], A, A)

    return J

def T_int(a1, d1, N1, a2, d2, N2, A):
    J = 0

    for i in range(0, 3):
        for j in range(0, 3):
            J = J + N1[i] * N2[j] * d1[i] * d2[j] * T_GTO(a1[i], a2[j], A, A)

    return J

def V_int(Z, a1, d1, N1, a2, d2, N2, A):
    J = 0

    for i in range(0, 3):
        for j in range(0, 3):
            J = J + N1[i] * N2[j] * d1[i] * d2[j] * V_GTO(Z, a1[i], a2[j], A, A)

    return J

def J_int(a1, d1, N1, a2, d2, N2, a3, d3, N3, a4, d4, N4, A):
    J = 0

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                for l in range(0, 3):
                    J = J + N1[i] * N2[j] * N3[k] * N4[l] * d1[i] * d2[j] * d3[k] * d4[l] * J_GTO(a1[i], a2[j], a3[k], a4[l], A, A, A, A)

    return J

def Sij(n, a, d, N, A):
    S = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            S[i, j] = S_int(a[i], d[i], N[i], a[j], d[j], N[j], A)

    return S

def Hij(Z, n, a, d, N, A):
    H = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            H[i, j] = T_int(a[i], d[i], N[i], a[j], d[j], N[j], A) + V_int(Z, a[i], d[i], N[i], a[j], d[j], N[j], A)

    return H

def Jijls(n, a, d, N, A):
    J = np.zeros((n, n, n, n))

    for i in range(0, n):
        for j in range(0, n):
            for l in range(0, n):
                for s in range(0, n):
                    J[i, j, l, s] = J_int(a[i], d[i], N[i], a[j], d[j], N[j], a[l], d[l], N[l], a[s], d[s], N[s], A)

    return J

def Gij(n, J, P):
    G = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            for l in range(0, n):
                for s in range(0, n):
                    G[i, j] = G[i, j] + P[l, s] * (J[i, j, l, s] - 0.5 * J[i, s, l, j])

    return G

def Pij(n, Z, C):
    P = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, int(Z/2)):
                P[i, j] = P[i, j] + 2 * C[i, k] * C[j, k]

    return P

def E(Z, P, epsilon, H):
    E = 0

    for i in range(0, int(Z/2)):
        E = E + epsilon[i]

    return E + 0.5 * (P * H).sum()

def scf(Z, a, d, N, A):
    n = int(Z/2)

    H = Hij(Z, n, a, d, N, A)
    J = Jijls(n, a, d, N, A)
    S = Sij(n, a, d, N, A)

    epsilon, C = sp.linalg.eigh(H, S)
    P0 = Pij(n, Z, C)
    E0 = E(Z, P0, epsilon, H)

    for i in range(0, 1000):
        G = Gij(n, J, P0)
        F = H + G

        epsilon, C = sp.linalg.eigh(F, S)
        P1 = Pij(n, Z, C)
        E1 = E(Z, P1, epsilon, H)

        if np.abs(E1 - E0) < 1e-14:
            return E1

        E0 = E1
        P0 = P1

    return E0

def helium_STO3g():
    a = [[0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]]
    d = [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]]

    N = [[np.float_power(2 * a[0][0]/np.pi, 0.75), np.float_power(2 * a[0][1]/np.pi, 0.75), np.float_power(2 * a[0][2]/np.pi, 0.75)]]
    A = np.zeros(3)

    ref = -(0.903561157603416 + 1.9998094077463793)

    return (a, d, N, A, ref)

def lithium_STO3g():
    a = [[0.1611957475E+02, 0.2936200663E+01, 0.7946504870E+00],
         [0.6362897469E+00, 0.1478600533E+00, 0.4808867840E-01]]
    d = [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
         [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00]]

    N = [[np.float_power(2 * a[0][0]/np.pi, 0.75), np.float_power(2 * a[0][1]/np.pi, 0.75), np.float_power(2 * a[0][2]/np.pi, 0.75)],
         [np.float_power(2 * a[1][0]/np.pi, 0.75), np.float_power(2 * a[1][1]/np.pi, 0.75), np.float_power(2 * a[1][2]/np.pi, 0.75)]]
    A = np.zeros(3)

    ref = -(0.19813367372815285 + 2.7796988931861444 + 4.500094877159027)

    return (a, d, N, A, ref)

def berylium_STO3g():
    a = [[0.3016787069E+02, 0.5495115306E+01, 0.1487192653E+01],
         [0.1314833110E+01, 0.3055389383E+00, 0.9937074560E-01]]
    d = [[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
         [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00]]

    N = [[np.float_power(2 * a[0][0]/np.pi, 0.75), np.float_power(2 * a[0][1]/np.pi, 0.75), np.float_power(2 * a[0][2]/np.pi, 0.75)],
         [np.float_power(2 * a[1][0]/np.pi, 0.75), np.float_power(2 * a[1][1]/np.pi, 0.75), np.float_power(2 * a[1][2]/np.pi, 0.75)]]
    A = np.zeros(3)

    ref = -(0.3426013831573885 + 0.6692439025523593 + 5.655569936730533 + 8.000989678081151)

    return (a, d, N, A, ref)

helium3g = helium_STO3g()
lithium3g = lithium_STO3g()
berylium3g = berylium_STO3g()

helium = scf(2, helium3g[0], helium3g[1], helium3g[2], helium3g[3])
lithium = scf(3, lithium3g[0], lithium3g[1], lithium3g[2], lithium3g[3])
berylium = scf(4, berylium3g[0], berylium3g[1], berylium3g[2], berylium3g[3])

helium_err = np.abs(np.abs(helium - helium3g[4])/helium) * 100
lithium_err = np.abs(np.abs(lithium - lithium3g[4])/lithium) * 100
berylium_err = np.abs(np.abs(berylium - berylium3g[4])/berylium) * 100

print("Helium energy: Measured: ", helium, " Experimental: ", helium3g[4], " Relative error: ", str(helium_err) + "%")
print("Lithium energy: Measured: ", lithium, " Experimental: ", lithium3g[4], " Relative error: ", str(lithium_err) + "%")
print("Berylium energy: Measured: ", berylium, " Experimental: ", berylium3g[4], " Relative error: ", str(berylium_err) + "%")