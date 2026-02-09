from firedrake import *
from firedrake.petsc import PETSc
optDB = PETSc.Options()
n = optDB.getInt("N", 8)
neig = optDB.getInt("neig", 10)


print("------!INFO!------")
print(f"N: {n}, neig: {neig}")
mesh = RectangleMesh(n,n, pi, pi, diagonal="crossed")
normal = FacetNormal(mesh)
tangent = as_vector((-normal[1], normal[0]))
V = VectorFunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(curl(u), curl(v))*dx # Here curl in equivaelent to curl bar in the Cambrdidge slides
a += 1e8*inner(dot(u, tangent), dot(v, tangent))*ds  # boundary condition term
b = inner(u, v)*dx
bc = []
#Solve the generalized eigenvalue problem
eigenproblem = LinearEigenproblem(A=a, M=b, bcs=bc)
opts = {"eps_gen_hermitian": None,
        "eps_smallest_real": None,
        "eps_target": 1,
        "eps_target_real": None,
        "st_type": "sinvert",
        "st_pc_factor_mat_solver_type": "mumps"
}
print("Solving ...")
eigensolver = LinearEigensolver(eigenproblem,
                                n_evals=neig,
                                solver_parameters=opts,
                                options_prefix="")
nconv = eigensolver.solve()
for k in range(nconv):
    lam = eigensolver.eigenvalue(k)
    print(f"{k}-th computed eigenvalue {lam:.8e}")

