from firedrake import *
import numpy as np
def norm(u):
    """
    This function computes the H1 norm of a function u.
    """
    return sqrt(assemble(inner(u, u) * dx)+assemble(inner(grad(u), grad(u)) * dx))
def LaplaceEigenvalues(N):
    """
    This function computes the first 10 eigenvalues of the Laplace operator
    on the unit interval [0, pi] with Dirichlet boundary conditions.
    """

    # Constructing a mesh for the unit interval
    mesh = SquareMesh(N, N, np.pi)

    # Defining the function space and the trial/test functions
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Defining the bilinear form associated with the Laplace operator,
    # the mass matrix and the boundary conditions
    a = inner(grad(u), grad(v)) * dx
    m = inner(u, v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    # Initializing an eigenvalue problem and solving for the first 10
    # eigenvalues
    eigenproblem = LinearEigenproblem(A=a, M=m, bcs=bc, restrict=True)
    opts = {"eps_gen_hermitian": None,
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
eigs, eigfs = LaplaceEigenvalues(16)
for i in range(10):
    print(f"Eigenvalue {i+1}: {eigs[i]:.6f}")

#We now compute the rate of convergence of the first 10 eigenvalues
#to do this we solve the eigenvalue problem for N = 16, 32, 64, 128, 256
exact_eigs = [2, 5, 5, 8, 10, 10, 13, 13, 17, 17]
eigs = []; eigfs = []
for ref in range(5):
    N = 2**(ref+4)
    eig, eigf = LaplaceEigenvalues(N)
    eigs.append([eig[i] for i in range(10)])
    eigfs.append([eigf[i] for i in range(10)])

#We compute the rate of convergence using polyfit
log_rate = []; log_ratef = []
for i in range(10):
    err = []
    for ref in range(3):
        err.append(abs(eigs[ref][i] - exact_eigs[i]))
    log_rate.append(np.polyfit(np.log([16, 32, 64]), np.log(err), 1)[0])
print("The rate of convergence of the first 10 eigenvalues of the Laplace operator on the unit interval [0, pi] is:")
for i, rate in enumerate(log_rate):
    print(f"Rate of convergence of the {i+1} eigenvalue: {-1*rate:.6f}")

#We now compute the rate of convergence of the eigenfunctions assosiated to the eigevalue 5
mesh = SquareMesh(128, 128, np.pi)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
exact_eigf_one = Function(V).interpolate(sin(2*x)*sin(1*y))
exact_eigf_two = Function(V).interpolate(sin(1*x)*sin(2*y))

err_f = []

for ref in range(3):
    projected_eigf_one = Function(V).interpolate(eigfs[ref][1][0])
    projected_eigf_two = Function(V).interpolate(eigfs[ref][2][0])
    
    rotate_one = assemble(inner(exact_eigf_one,projected_eigf_one)*dx)
    rotate_two = assemble(inner(exact_eigf_one,projected_eigf_two)*dx)
    err_one = norm(rotate_one*projected_eigf_one+rotate_two*projected_eigf_two- exact_eigf_one)
    rotate_one = assemble(inner(exact_eigf_two,projected_eigf_one)*dx)
    rotate_two = assemble(inner(exact_eigf_two,projected_eigf_two)*dx)
    err_two = norm(rotate_one*projected_eigf_one+rotate_two*projected_eigf_two- exact_eigf_two)

    max_err = max(err_one, err_two)
    err_f.append(max_err)
log_ratef = np.polyfit(np.log([16, 32, 64]), np.log(err_f), 1)[0]
print(f"Rate of convergence of the eigenfunction assosciated to the multiple eigenvalue 5: {-1*log_ratef:.6f}")

