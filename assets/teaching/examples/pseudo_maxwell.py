"""
python -i pseudo_maxwell_schur.py -pseudo_svd_type trlanczos -pseudo_st_shift 0.1 -pseudo_svd_max_it 10000 -pseudo_st_type sinvert -pseudo_st_pc_type lu -pseudo_eps_target 0.1 -npts 2000 -Re 1 -wind 3 -invert_stencil
"""
from firedrake import *
import numpy as np
import sys
import time
import slepc4py
import petsc4py
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
petsc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc

opt = PETSc.Options()
Re = opt.getReal("Re", 1.0)     # Reynolds number       
Re = Constant(Re)

self_adjoint = opt.getBool("self_adjoint", False)     # Invert stencil     
neig = opt.getInt("neig", 10)     # Number of eigenvalue to compute
npts = opt.getInt("npts", 100)
wind_type = opt.getInt("wind", 1)
pdir = opt.getString("periodic_direction", "both")     # Reynolds number       
pseudo_plot = opt.getBool("pseudo_plot", True)     # Plot pseudo-spectra
radius_cap = opt.getReal("radius_cap", 1e10)     # Cap on radius for plotting eigenvalues
invert_stencil = opt.getBool("invert_stencil", False)     # Invert stencil

print("-------------------------|Parameters|-------------------------")
print(f"Reynolds number: {float(Re):.2e}")
print(f"Invert stencil: {invert_stencil}")
print(f"Self-adjoint advection: {self_adjoint}")
print(f"Number of eigenvalues to compute: {neig}")
print(f"Number of points for pseudo-spectra: {npts}")
print(f"Wind type: {wind_type}")
print(f"Periodic direction: {pdir}")
print(f"Radius cap for plotting eigenvalues: {radius_cap:.2e}")
print(f"Plot pseudo-spectra: {pseudo_plot}")
print("--------------------------------------------------------------")

slepc4py.init(sys.argv)

mesh = PeriodicRectangleMesh(32, 32, np.pi, np.pi, direction=pdir) 
x,y  = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "N1curl", 1)
Q = FunctionSpace(mesh, "CG", 1)
X = MixedFunctionSpace([V, Q])
u,psi = TrialFunctions(X)
v,phi = TestFunctions(X)

if wind_type==0:
    self_adjoint = True
elif wind_type==1:
    w = as_vector([1,1])
elif wind_type==2:
    w = as_vector([2*cos(2*x)*sin(2*y),2*sin(2*x)*cos(2*y)])
elif wind_type==3:
    w = as_vector([sin(y), sin(x)])
else:
    raise ValueError("Unknown wind type")


def cross_2d(w,u):
    return w[0]*u[1]-w[1]*u[0]

a = (1/Re)*inner(curl(u), curl(v))*dx
if self_adjoint:
    pass
else:
    a += inner(cross_2d(w,u), curl(v))*dx

a += inner(grad(psi), v)*dx + inner(u, grad(phi))*dx
m = inner(u, v)*dx 

A = assemble(a)
M = assemble(m)

if invert_stencil:
    eigenproblem = LinearEigenproblem(A=m, M=a) 
else:
    eigenproblem = LinearEigenproblem(A=a, M=m)
opts = {"eps_gen_non_hermitian": None,
        "eps_largest_real": None,
        "eps_target": 0.25/float(Re),
        "eps_target_real": None,
        "st_type": "sinvert",
        "st_pc_type": "qr",
        #"st_pc_factor_mat_solver_type": "mumps"
}
eigensolver = LinearEigensolver(eigenproblem, n_evals=neig, solver_parameters=opts)
nconv = eigensolver.solve()
lamhs = []
for k in range(neig):
    if invert_stencil:
        lam = 1/eigensolver.eigenvalue(k)
    else:
        lam = eigensolver.eigenvalue(k)
    print(f"{k}-th computed eigenvalue {lam:.2e}")
    lamhs = lamhs + [-lam]

print("Contour of the interpolant")
import pseudo_tool
opt = {"figure_title": f"Dynamo Pseudo-Spectra Re={float(Re):.1e}, $u_{wind_type}$, Edge",
       "flip": True}
pseudo_tool.draw_pseudo_spectra(A, M, (-4, 20, -12, 12), lams=lamhs, npts=2000, opts=opt)
