import scipy as sp
from scipy import constants as const
import numpy as np
import mesh_generation as mg
import matplotlib.pyplot as plt
import perturbation_theory as pt

#This function generates the element connectivity matrix
#N -> integer, number of nodes in the mesh
def LTOG(N):
    T = np.zeros((N - 1, 2))
    
    e = 0
    
    for i in range(1, N):
        T[e][0] = i - 1
        T[e][1] = i
        e = e + 1
        
    return T

#This function transforms the element local coordinate t into the global coordinate x
#t -> double, local coordinate
#e -> integer, element index
#h -> double, mesh spacing
#nodes -> array, all nodes in the mesh
#LTOG -> matrix, element connectivity matrix
def to_global(t, e, h, nodes, LTOG):
    i0 = int(LTOG[e, 0])
    i1 = int(LTOG[e, 1])

    return 0.5 * h * t + 0.5 * (nodes[i0] + nodes[i1])

def to_local(x, e, h, nodes, LTOG):
    i0 = int(LTOG[e, 0])
    i1 = int(LTOG[e, 1])
    
    return 2/h * x - (nodes[i0] + nodes[i1]) / h

#This function calculates the shape function in a point for a specific node in an element
#t -> double, local coordinate within the given element
#i -> integer, target node index
def S(t, i):
    if i == 0:
        return 0.5 * (1 - t)
    
    return 0.5 * (1 + t)

#This function calculates the shape function's first derivative in a point for a specific node in an element
#t -> double, local coordinate within the given element
#i -> integer, target node index
def dS(t, i):
    if i == 0:
        return -0.5
    
    return 0.5

#This function calculates the elemental stiffness matrix component
#i -> integer, row index
#j -> integer, column index
#h -> double, mesh spacing
def K(i, j, h):
    return h * S(0, i) * S(0, j)

#This function calculates the first order elemental stiffness matrix component
#i -> integer, row index
#j -> integer, column index
#h -> double, mesh spacing
def K1(i, j, h):
    return 2 * S(0, i) * dS(0, j)

#This function calculates the second order elemental stiffness matrix component
#i -> integer, row index
#j -> integer, column index
#h -> double, mesh spacing
def K2(i, j, h):
    return -4/h * dS(0, i) * dS(0, j)

#This function solves a general linear eigenvalue ODE with variable coefficients with homogenous essential BCs using the Finite Element Method
#equation -> array of function handles, represents coefficients for u'', u', u in that order
#domain -> tuple, represents the boundary of the domain of interest
#N -> integer, number of nodes in the finite element method's mesh
def solve(equation, domain, N):
    mesh = mg.generate_mesh_1D(N, domain)
    nodes = mesh[0]
    h = mesh[1]
    
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    T = LTOG(N)
    
    for e in range(0, N - 1):
        for i in range(0, 2):
            for j in range(0, 2):
                I = int(T[e][i])
                J = int(T[e][j])
                x = to_global(0, e, h, nodes, T)

                A[I][J] = A[I][J] + equation[0](x) * K2(i, j, h) + equation[1](x) * K1(i, j, h) + equation[2](x) * K(i, j, h)
                B[I][J] = B[I][J] + K(i, j, h)
         
    H = (A, B)
    A = np.delete(A, N - 1, 0)
    A = np.delete(A, N - 1, 1)
    A = np.delete(A, 0, 0)
    A = np.delete(A, 0, 1)
    
    B = np.delete(B, N - 1, 0)
    B = np.delete(B, N - 1, 1)
    B = np.delete(B, 0, 0)
    B = np.delete(B, 0, 1)
    
    (E, v) = sp.linalg.eigh(A, B, subset_by_index=[0, 10], driver="gvx")
    E = np.atleast_2d(E).transpose()
    E = np.divide(E, E[0])
    
    u = []
    
    for i in range(0, len(E)):
        psi = []
        psi = np.append(psi, v[:, i])
        psi = np.append(psi, 0)
        psi = np.insert(psi, 0, 0)
       
        u.append(psi)

    return (E, u)