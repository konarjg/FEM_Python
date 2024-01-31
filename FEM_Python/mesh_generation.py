import scipy as sp
import numpy as np

#This function generates the mesh for finite element method solution in one dimension
#N -> integer, number of nodes in the mesh
#domain -> tuple, target domain boundary
def generate_mesh_1D(N, domain):
    mesh = []
    x = np.linspace(domain[0], domain[1], num=N, retstep=True)

    for i in range(0, N):
        mesh.append(x[0][i])
        
    return (mesh, x[1])

#This function generates the mesh for finite element method solution in two dimensions using rectangles
#Nx -> integer, number of nodes per horizontal direction
#Ny -> integer, number of nodes per vertical direction
#domain_x -> tuple, horizontal domain boundary
#domain_y -> tuple, vertical domain boundary
def generate_mesh_2D(Nx, Ny, domain_x, domain_y):
    mesh = []

    x = np.linspace(domain_x[0], domain_x[1], num=Nx, retStep = True)
    y = np.linspace(domain_y[0], domain_y[1], num=Ny, retStep = True)

    for i in range(0, Nx):
        for j in range(0, Ny):
            mesh.append((x[0][i], y[0][j]))

    return (mesh, x[1], y[1])
