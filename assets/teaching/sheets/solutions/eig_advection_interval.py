
from firedrake import *
import numpy as np

eta = Constant(0.1)

def norm(u):
    """
    This function computes the H1 norm of a function u.
    """
    return sqrt(assemble(inner(u, u) * dx)+assemble(inner(grad(u), grad(u)) * dx))
def AdvectionDiffusionEigenvalues(N):
    """
    This function computes the first 10 eigenvalues of the Laplace operator
    on the unit interval [0, pi] with Dirichlet boundary conditions.
    """

    # Constructing a mesh for the unit interval
    mesh = IntervalMesh(N, np.pi)

    # Defining the function space and the trial/test functions
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Defining the bilinear form associated with the Laplace operator,
    # the mass matrix and the boundary conditions
    a = eta*inner(grad(u), grad(v)) * dx - inner(grad(u)[0], v) * dx
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

print("The first 10 eigenvalues of the Laplace operator on the unit interval [0, pi] are:")
eigs, eigfs = AdvectionDiffusionEigenvalues(16)
for i in range(10):
    print(f"Eigenvalue {i+1}: {eigs[i]:.6f}")

#We now compute the rate of convergence of the first 10 eigenvalues
#to do this we solve the eigenvalue problem for N = 16, 32, 64, 128
exact_eigs = [ 0.25/float(eta) + float(eta)*(k+1)**2 for k in range(10)]
eigs = []; eigfs = []
for ref in range(3):
    N = 2**(ref+5)
    eig, eigf = AdvectionDiffusionEigenvalues(N)
    eigs.append([eig[i] for i in range(10)])
    eigfs.append([eigf[i] for i in range(10)])
#We compute the rate of convergence using polyfit
log_rate = []; log_ratef = []
for i in range(10):
    err = []
    errf = []
    for ref in range(3):
        err.append(abs(eigs[ref][i] - exact_eigs[i]))
        ref_mesh = IntervalMesh(128, np.pi); x = SpatialCoordinate(ref_mesh)
        V = FunctionSpace(ref_mesh, "CG", 2)
        exact_eigf = Function(V).interpolate(exp(-0.5*x[0]/eta)*sin((i+1)*x[0]))
        #We project on the fine mesh the real part of the eigenfunction
        projected_eigf = Function(V).interpolate(eigfs[ref][i][0])
        rotate = assemble(inner(exact_eigf,projected_eigf)*dx)/assemble(inner(projected_eigf,projected_eigf)*dx)
        errf.append(norm(rotate*projected_eigf- exact_eigf))
    log_rate.append(np.polyfit(np.log([32, 64, 128]), np.log(err), 1)[0])
    log_ratef.append(np.polyfit(np.log([32, 64, 128]), np.log(errf), 1)[0])
print("The rate of convergence of the first 10 eigenvalues of the Laplace operator on the unit interval [0, pi] is:")
for i, rate in enumerate(log_rate):
    print(f"Rate of convergence of the {i+1} eigenvalue: {-1*rate:.6f}")
for i, ratef in enumerate(log_ratef):
    print(f"Rate of convergence of the {i+1} eigenfunction: {-1*ratef:.6f}")