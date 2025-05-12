from firedrake import *
import numpy as np
from netgen.geom2d import SplineGeometry

def norm(u):
    """
    This function computes the H1 norm of a function u.
    """
    return sqrt(assemble(inner(u, u) * dx)+assemble(inner(grad(u), grad(u)) * dx))
def LaplaceEigenvalues(geo, N):
    """
    This function computes the first 10 eigenvalues of the Laplace operator
    on the unit interval [0, pi] with Dirichlet boundary conditions.
    """

    # Constructing a mesh for the unit interval
    ngmesh = geo.GenerateMesh(maxh=1.0/(N**2))
    mesh = Mesh(ngmesh)

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

geo = SplineGeometry()
pnts = [(0, 0), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0),
        (-1, -1), (0, -1)]
p1, p2, p3, p4, p5, p6, p7, p8 = [geo.AppendPoint(*pnt) for pnt in pnts]
curves = [[["line", p1, p2], "line"],
          [["spline3", p2, p3, p4], "curve"],
          [["spline3", p4, p5, p6], "curve"],
          [["spline3", p6, p7, p8], "curve"],
          [["line", p8, p1], "line"]]
[geo.Append(c, bc=bc) for c, bc in curves]

exact = 3.375610652693620492628**2
err = []
H = []
for ref in range(3):
    N = 2**(ref+2)
    H = H + [1.0/(N**2)]
    eigs, eigfs = LaplaceEigenvalues(geo, N)
    print(f"Eigenvalues for N={N}: {eigs}")
    err.append(abs(eigs[0]-exact))
# We compute the rate of convergence using polyfit
rate = np.polyfit(np.log(H), np.log(err), 1)[0]
print(f"Rate of convergence: {rate:.2f}")