import scipy as np

#This function generates the mesh for finite element method solution in one dimension
#N -> integer, number of nodes in the mesh
#domain -> tuple, target domain boundary
def generate_mesh_1D(N, domain):
    mesh = []
    x = np.linspace(domain[0], domain[1], num=N, retstep=True)

    for i in range(0, N):
        mesh.append(x[0][i])
        
    return (mesh, x[1])
