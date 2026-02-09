from firedrake import *
import slepc4py
import petsc4py
import sys
petsc4py.init(sys.argv)
slepc4py.init(sys.argv)
from slepc4py import SLEPc
from petsc4py import PETSc
import time
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from tqdm import tqdm


def compute_residual(z, A, B, nsvd=3):
        svd = SLEPc.SVD()
        svd.create()
        Bs = B.duplicate(copy=True)
        Bs.scale(-z)
        C = A.duplicate
        C = A+Bs
        svd.setOperator(C, B)
        svd.setDimensions(nsvd)
        svd.setWhichSingularTriplets(SLEPc.SVD.Which.SMALLEST)  # focus on smallest singular values
        svd.setOptionsPrefix("pseudo_")
        svd.setFromOptions()
        #pdb.set_trace()
        svd.solve()
        
        # Get the number of converged singular values
        nconv = svd.getConverged()
        if nconv > 0:
        # Get the smallest singular value
            sval = svd.getValue(0)
            return sval
        else:
            raise RuntimeError("SVD did not converge")
            return 0.0
def compute_projections(A, M, n_subspace=40, opts=None):
    tic = time.time()
    eps = SLEPc.EPS().create()
    eps.setOperators(A.petscmat, M.petscmat)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(n_subspace)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    eps.setTwoSided(True) #Two sided converge with correct order for NHP (observation by Yuji!)
    eps.setOptionsPrefix("pseudo_")
    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()
    k = min(n_subspace, nconv)
    # Collect the first k right/left eigenvectors into PETSc Vecs
    Vr, Vi = PETSc.Vec().createMPI(A.petscmat.getSizes()[0]), PETSc.Vec().createMPI(A.petscmat.getSizes()[0])
    Wr, Wi = PETSc.Vec().createMPI(A.petscmat.getSizes()[0]), PETSc.Vec().createMPI(A.petscmat.getSizes()[0])

    Vcols, Wcols = [], []
    for i in range(k):
        eps.getEigenpair(i, Vr, Vi)       # right eigenvector (real/imag parts if complex build uses split)
        Vcols.append(Vr.copy())
        eps.getLeftEigenvector(i, Wr)     # left eigenvector i
        Wcols.append(Wr.copy())
    #Project the pencil to the subspace
    Vbv = SLEPc.BV().create(); Vbv.setSizesFromVec(Vcols[-1], n_subspace)
    Wbv = SLEPc.BV().create(); Wbv.setSizesFromVec(Wcols[-1], n_subspace)
    Vbv.setFromOptions(); Wbv.setFromOptions()

    for j,v in enumerate(Vcols): Vbv.insertVec(j, v)
    for j,w in enumerate(Wcols): Wbv.insertVec(j, w)
    Vbv.orthogonalize()    # optional but helpful
    Wbv.orthogonalize()

    # Ar = W^H A V,  Br = W^H B V
    Ar = PETSc.Mat().createDense([k,k]); Ar.setUp()
    Br = PETSc.Mat().createDense([k,k]); Br.setUp()
    Ar = Vbv.matProject(A.petscmat, Wbv)   # oblique projection Y^H*A*X
    Br = Vbv.matProject(M.petscmat, Wbv)   # oblique projection Y^H*A*X
    Ar.assemble(); Br.assemble()
    toc = time.time()
    print(f"Time taken to do Schur decomposition: {toc-tic}")
    return Ar, Br
def draw_pseudo_spectra(A,M, rect, npts=1000, opts={}, lams=None):
    print("Computing projections...")
    Ar, Br = compute_projections(A,M, opts=opts)
    def R(z, pbar) :
        pbar.update(1)
        return compute_residual(z, Ar, Br)
    R = np.vectorize(R)

    print("Contour of the interpolant")
    tic = time.time()
    minX, maxX, minY, maxY = rect
    x = np.random.uniform(minX, maxX, npts)
    y = np.random.uniform(minY, maxY, npts)
    with tqdm(total=len(x)) as pbar:
        z = R(x+y*1j, pbar)
    toc = time.time()
    if "flip" in opts and opts["flip"]:
        x = -x
        y = -y
    tri = Triangulation(x, y)
    plt.figure(figsize=(6,5))
    cs = plt.tricontour(tri, z, colors='k', linewidths=0.4)  # optional line contours
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    csf = plt.tricontourf(tri, z, levels=50, cmap='plasma')
    plt.colorbar(csf)
    if lams is not None:
        plt.scatter([lam.real for lam in lams], [lam.imag for lam in lams], color='white')
    print(f"Time taken to evaluate pseudo-spectra for plot: {toc-tic}")
    plt.title(opts["figure_title"] if "figure_title" in opts else "Pseudo-spectra")
    plt.show()
