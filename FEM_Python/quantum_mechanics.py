import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import clr

clr.AddReference("System")
from System.Numerics import Complex

#This function calculates the normalization constant for given wavefunction in cartesian coordinates
#psi -> vector, vector of wavefunction values at grid points
def normalize(psi, x):
    return np.sqrt(1/(np.sum(psi**2) * (x[1] - x[0])))

#This function calculates the normalization constant for given wavefunction in spherical coordinates
#psi -> vector, vector of wavefunction values at grid points
def normalize_spherical(psi, x):
    return np.sqrt(1/(4 * np.pi * np.sum(psi**2 * x**2) * (x[1] - x[0])))

def momentum(psi, x, p, spherical = False):
    u = []

    for j in range(0, len(psi)):
        def f(k):
            K = np.exp(-1j * k * x)

            if spherical:
                return 1/np.sqrt(2 * np.pi) * (K * psi[j] * x**2).sum() * (x[1] - x[0])

            return 1/np.sqrt(2 * np.pi) * (K * psi[j]).sum() * (x[1] - x[0])

        F = np.zeros(len(x), dtype=np.complex64)

        for i in range(0, len(p)):
            F[i] = f(p[i])

        if spherical:
            F = normalize_spherical(F, x) * F
        else:
            F = normalize(F, p) * F

        T = []

        for i in range(0, len(p)):
            T.append(Complex(float(np.real(F[i])), float(np.imag(F[i]))))

        u.append(T)

    return u

#This function calculates the Hamiltonian matrix for given Schrodinger's equation coefficients
#a -> array of functions, coefficients in the Schrodinger's equation
#x -> vector, solution grid
#homogeneous -> boolean, wether homogeneous Dirichlet boundary conditions must be specified
def Hij(a, x, homogeneous=True):
    n = len(x)
    h = x[1] - x[0]

    H = np.zeros((n, n))

    for i in range(0, n):
        if i - 1 >= 0:
            H[i - 1, i] = H[i, i - 1] = a[0](x[i])/h**2 - a[1](x[i])/(2 * h)

        if i + 1 < n:
            H[i + 1, i] = H[i, i + 1] = a[0](x[i])/h**2 + a[1](x[i])/(2 * h)

        H[i, i] = -2 * a[0](x[i])/(h**2) + a[2](x[i])

    if homogeneous:
        H = np.delete(H, n - 1, 0)
        H = np.delete(H, n - 1, 1)
        H = np.delete(H, 0, 0)
        H = np.delete(H, 0, 1)

    return H

#This function solves the time independent one dimensional Schrodinger's equation with given coefficients
#equation -> array of functions, coefficients in the TISE in order u'', u', u
#domain -> tuple, boundary of the domain
#n -> integer, number of grid points
#homogeneous -> boolean, wether homogeneous Dirichlet boundary conditions must be specified
#spherical -> boolean, wether spherical coordinates should be assumed
#returns tuple of energies, normalized bound states and grid
def solve(equation, domain, n, homogeneous=True, spherical=False):
    x = np.linspace(domain[0], domain[1], n)
    H = Hij(equation, x, homogeneous)

    E, u = sp.linalg.eigh(H, subset_by_index=[0, 2])

    psi = []

    for i in range(0, len(E)):
        psi_i = u[:, i]
        
        if homogeneous:
            psi_i = np.append(psi_i, 0)
            psi_i = np.insert(psi_i, 0, 0)

        N = normalize(psi_i, x)

        if spherical:
            N = normalize_spherical(psi_i, x)

        psi.append(N * psi_i)

    return (E * 27.2, psi, x)

#This function solves the 1d infinite rectangular well quantum problem
#L -> float, width of the box
#n -> int, node count
#returns tuple of energy eigenvalues, eigenstates and grid of points
def infinite_rect_well(L, n):
    a = [lambda x: -0.5, lambda x: 0, lambda x: 0]
    return solve(a, (0, L), n)

#This function solves the finite rectangular well quantum problem
#V0 -> float, potential barrier height
#L -> float, width of the box
#n -> int, node count
#returns tuple of energy eigenvalues, eigenstates and grid of points
def finite_rect_well(V0, L, n):
    def V(x, V0, L):
        if -L/2 < x < L/2:
            return 0

        return V0

    a = [lambda x: -0.5, lambda x: 0, lambda x: V(x, V0, L)]
    return solve(a, (-L/2 - 5, L/2 + 5), n, False)

#This function solves the 1d quantum harmonic oscillator problem
#n -> int, node count
#returns tuple of energy eigenvalues, eigenstates and grid of points
def harmonic_oscillator(n):
    a = [lambda x: -0.5, lambda x: 0, lambda x: 0.5 * x**2]
    return solve(a, (-5, 5), n)

#This function solves the delta potential well problem
#n -> int, node count
#returns tuple of energy eigenvalues, eigenstates, grid of points
def delta_potential(n):
    def V(x, h):
        if np.abs(x) <= h:
            return 200

        return 0 

    a = [lambda x: -0.5, lambda x: 0, lambda x: -5 * V(x, 2/(n - 1))]
    return solve(a, (-1, 1), n, False)

#This function solves the radial equation of Hydrogen atom
#n -> int, node count
#l -> int, azimuthal quantum number
#returns tuple of energy eigenvalues, eigenstates, grid of points
def hydrogen_atom(n, l):
    a = [lambda x: -0.5, lambda x: 0, lambda x: l * (l + 1)/(2 * x**2 + 0.0001) - 1/(x + 0.0001)]
    return solve(a, (0, 200), n, True, True)

#This function solves the 2d infinite rectangular well quantum problem
#Lx, Ly -> floats, width and height of the box
#N -> int, node count per direction
#returns tuple of energy eigenvalues, eigenstates and grid of points
def infinite_rect_well_2d(N, Lx, Ly):
    Ex, ux, x = infinite_rect_well(Lx, N)
    Ey, uy, y = infinite_rect_well(Ly, N)

    psi = []
    E = []

    for i in range(0, len(Ex)):
        for j in range(0, len(Ey)):
            E.append(Ex[i] + Ey[j])
            psi.append(np.outer(ux[i], uy[j]))

    solutions = zip(E, psi)
    result = sorted(solutions, key = lambda x: x[0])
            
    E, psi = zip(*result)
    return E, psi, ux, uy, x, y

#This function solves the 2d finite rectangular well quantum problem
#V0 -> float, height of the potential barriers
#Lx, Ly -> floats, width and height of the box
#N -> int, node count per direction
#returns tuple of energy eigenvalues, eigenstates and grid of points
def finite_rect_well_2d(N, V0, Lx, Ly):
    Ex, ux, x = finite_rect_well(V0, Lx, N)
    Ey, uy, y = finite_rect_well(V0, Ly, N)
    psi = []
    E = []

    for i in range(0, len(Ex)):
        for j in range(0, len(Ey)):
            E.append(Ex[i] + Ey[j])
            psi.append(np.outer(ux[i], uy[j]))

    solutions = zip(E, psi)
    result = sorted(solutions, key = lambda x: x[0])
            
    E, psi = zip(*result)
    return E, psi, ux, uy, x, y

#This function solves the 2d harmonic oscillator quantum problem
#N -> int, node count per direction
#returns tuple of energy eigenvalues, eigenstates and grid of points
def harmonic_oscillator_2d(N):
    Ex, ux, x = harmonic_oscillator(N)
    Ey, uy, y = harmonic_oscillator(N)
    psi = []
    E = []

    for i in range(0, len(Ex)):
        for j in range(0, len(Ey)):
            E.append(Ex[i] + Ey[j])
            psi.append(np.outer(ux[i], uy[j]))

    solutions = zip(E, psi)
    result = sorted(solutions, key = lambda x: x[0])
            
    E, psi = zip(*result)
    return E, psi, ux, uy, x, y

#This function solves the 2d delta potential quantum problem
#N -> int, node count per direction
#returns tuple of energy eigenvalues, eigenstates and grid of points
def delta_potential_2d(N):
    Ex, ux, x = delta_potential(N)
    Ey, uy, y = delta_potential(N)

    psi = []
    E = []

    for i in range(0, len(Ex)):
        for j in range(0, len(Ey)):
            E.append(Ex[i] + Ey[j])
            psi.append(np.outer(ux[i], uy[j]))

    solutions = zip(E, psi)
    result = sorted(solutions, key = lambda x: x[0])
            
    E, psi = zip(*result)
    return E, psi, ux, uy, x, y

#This function solves the full equation of Hydrogen atom and returns the 2d projection
#N -> int, node count
#l -> int, azimuthal quantum number
#returns tuple of energy eigenvalues, eigenstates, grid of points
def hydrogen_atom2d(N):
    def Ylm(l, m, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(z/r)
        Y = sp.special.sph_harm(m, l, phi, theta)

        if m == 0:
            return np.real(Y)

        return np.sqrt(2) * np.real(Y)

    def u(x, y, z, l, m, psi_r):
        r0 = np.sqrt(x**2 + y**2 + z**2)
        return np.interp(r0, r, psi_r) * Ylm(l, m, x, y, z)

    E = []
    psi = []
    x = np.linspace(-200, 200, N)
    z = np.linspace(-200, 200, N)
    y = 1

    X, Z = np.meshgrid(x, z)

    i = 0
    u_r = []

    for l in range(0, 4):
        El, psi_r, r = hydrogen_atom(N, l)

        for k in range(0, len(El)):
            for m in range(-l, l + 1):    
                E.append(El[k])
                psi.append(u(X, y, Z, l, m, psi_r[k]))
                u_r.append(psi_r[k])

    solutions = zip(E, psi, u_r)
    result = sorted(solutions, key = lambda x: x[0])
    E, psi, u_r = zip(*result)

    return E, psi, u_r, x, z

#This function solves the integer quantum Hall effect equation returns the 2d projection
#B -> float, magnetic field strength
#Lx, Ly, Lz -> floats, width, depth and height of the semiconductor sample
#n -> int, node count per direction
#returns tuple of energy eigenvalues, eigenstates and grid of points
def hall_effect(n, B, Lx, Ly, Lz):
    l = 1/B
    
    ny = 1
    k = ny * 2 * np.pi / Ly

    Ez, psi_z, z = infinite_rect_well(Lz, n)
    a_x = [lambda x: -0.5, lambda x: 0, lambda x: 0.5 * B**2 * (x - l**2 * k)**2]
    Exy, psi_x, x = solve(a_x, (0, Lx), n)

    y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    
    E = []
    psi = []

    for nx in range(0, len(Exy)):
        ux = lambda x0: np.interp(x0, x, psi_x[nx])

        for nz in range(0, len(Ez)):
            uz = lambda z0: np.interp(z0, z, psi_z[nz])
            f = lambda x0, y0, z0: np.sqrt(1/Ly) * ux(x0) * uz(z0) * (y0/y0)

            E.append(Exy[nx] + Ez[nz])
            psi.append(f(X, Y, z))

    solutions = zip(E, psi)
    result = sorted(solutions, key = lambda x: x[0])
    E, psi = zip(*result)

    return E, psi, x, y