
from firedrake import *
import numpy as np

eta = Constant(0.5)
w = as_vector([1.0, 0.0])  # Advection velocity

def AdvectionDiffusionEigenvalues(N):
    """
    This function computes the first 10 eigenvalues of the Laplace operator
    on the unit interval [0, pi] with Dirichlet boundary conditions.
    """

    # Constructing a mesh for the unit interval
    mesh = SquareMesh(N, N, np.pi, np.pi)

    # Defining the function space and the trial/test functions
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Defining the bilinear form associated with the Laplace operator,
    # the mass matrix and the boundary conditions
    a = eta*inner(grad(u), grad(v)) * dx - inner(dot(w,grad(u)), v) * dx
    m = inner(u, v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    # Initializing an eigenvalue problem and solving for the first 10
    # eigenvalues
    eigenproblem = LinearEigenproblem(A=a, M=m, bcs=bc, restrict=True)
    opts = {"eps_gen_non_hermitian": None,
            "eps_smallest_real": None,
            "eps_type": "krylovschur",
            "st_type": "sinvert"}
    eigensolver = LinearEigensolver(eigenproblem, n_evals=10,
                                    solver_parameters=opts)
    nconv = eigensolver.solve()
    if nconv < 10:
        raise RuntimeError("The mesh is too coarse to compute the first 10 eigenvalues.")
    return ([eigensolver.eigenvalue(i) for i in range(10)],
            [eigensolver.eigenfunction(i) for i in range(10)])

# We compute the first 10 eigenvalues of the Laplace operator

print("The first 10 eigenvalues of the Laplace operator on the square [0, pi]*[0, pi] are:")
eigs, eigfs = AdvectionDiffusionEigenvalues(16)
for i in range(10):
    print(f"Eigenvalue {i+1}: {eigs[i]:.6f}")

#Saving the first 10 eigenfunctions to files
fp = File("output/eigenfunctions.pvd")
mesh = SquareMesh(16, 16, np.pi, np.pi)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
for i in range(10):
    u.interpolate(eigfs[i][0])
    fp.write(u, time=i)