from firedrake import *
from firedrake.petsc import PETSc
optDB = PETSc.Options()
n = optDB.getInt("N", 8)
neig = optDB.getInt("neig", 10)
print("------!INFO!------")
print(f"N: {n}, neig: {neig}")
mesh = RectangleMesh(n,n, pi, pi, quadrilateral=True)
normal = FacetNormal(mesh)

V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "DG", 0)
W = V*Q

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
a += 1e8 * inner(inner(sigma, normal),inner(tau, normal)) * ds  # boundary condition term
b = -inner(u,v) * dx

bc = []
#Solve the generalized eigenvalue problem
eigenproblem = LinearEigenproblem(A=a, M=b, bcs=bc)
opts = {"eps_gen_non_hermitian": None,
        "eps_smallest_real": None,
        "eps_target": 1,
        "eps_target_real": None,
        "st_type": "sinvert",
        "st_ksp_type": "gmres",
        "st_pc_type": "qr"
}
print("Solving ...")
eigensolver = LinearEigensolver(eigenproblem,
                                n_evals=neig,
                                solver_parameters=opts,
                                options_prefix="")
lamhs = []
nconv = eigensolver.solve()
for k in range(nconv):
    lam = eigensolver.eigenvalue(k)
    print(f"{k}-th computed eigenvalue {lam:.8e}")
    lamhs = lamhs + [lam]

