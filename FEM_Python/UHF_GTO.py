import math
import scipy as sp
import numpy as np
import sympy as sm
from sympy.printing import pprint
from sympy import oo

def dist(A, B):
    return np.linalg.norm(A - B)

def Fm(m, x):
    f = lambda t: t**(2 * m) * np.exp(-x * t**2)
    return sp.integrate.quad(f, 0, 1)[0]

def overlap(a, b, l1, l2, A, B):
    sp = 1/(2 * a + 2 * b)
    Up = np.sqrt(8 * np.pi**3) * sp**(l1 + l2 + 1.5) * np.exp(-2 * a * b * sp * dist(A, B)**2)
    P = (2 * a * A + 2 * b * B)*sp

    

def coulomb(a, b, c, d, l1, l2, l3, l4, A, B, C, D):
    sp = 1/(2 * a + 2 * b)
    Up = np.sqrt(8 * np.pi**3) * sp**(l1 + l2 + 1.5) * np.exp(-2 * a * b * sp * dist(A, B)**2)
    P = (2 * a * A + 2 * b * B)*sp

    sq = 1/(2 * c + 2 * d)
    Uq = np.sqrt(8 * np.pi**3) * sp**(l3 + l4 + 1.5) * np.exp(-2 * c * d * sq * dist(C, D)**2)
    Q = (2 * c * C + 2 * d * D)*sq

    R = Q - P
    R2 = dist(R, np.zeros(3))**2
    o2 = 1/(2 * sp + 2 * sq)
    T = o2 * R2
    U = Up * Uq
    m = l1 + l2 + l3 + l4

    N = (2 * a / np.pi)**0.75 * (2 * b / np.pi)**0.75 * (2 * c / np.pi)**0.75 * (2 * d / np.pi)**0.75

    return N * U * (2 * o2)**(m + 0.5) * np.sqrt(2/np.pi) * Fm(m, T)

