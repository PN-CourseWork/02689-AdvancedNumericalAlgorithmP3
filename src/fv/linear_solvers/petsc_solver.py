"""PETSc-based linear solver with HYPRE preconditioner and KSP reuse."""

import numpy as np
from scipy.sparse import csr_matrix
from petsc4py import PETSc


def petsc_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    ksp=None,
    tolerance=1e-6,
    max_iterations=1000,
    solver_type="bcgs",
    preconditioner="gamg",
    remove_nullspace=False,
):
    """Solve A x = b using PETSc with optional KSP reuse.

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format.
    b_np : np.ndarray
        Right-hand side vector.
    ksp : PETSc.KSP, optional
        Reusable KSP solver object. If None, a new KSP is created.
    tolerance : float, optional
        Convergence tolerance for the solver (default: 1e-6).
    max_iterations : int, optional
        Maximum number of iterations for the solver (default: 1000).
    solver_type : str, optional
        Type of PETSc solver to use (default: "bcgs").
    preconditioner : str, optional
        Type of PETSc preconditioner to use (default: "hypre").
    remove_nullspace : bool, optional
        Whether to handle the nullspace (default: False).
        If True, creates a constant nullspace vector and removes it from the RHS.

    Returns
    -------
    x_np : np.ndarray
        Solution vector x.
    ksp : PETSc.KSP
        KSP solver (returned for reuse).
    """
    n = A_csr.shape[0]

    # Create PETSc matrix from SciPy CSR
    A_petsc = PETSc.Mat().createAIJ(
        size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data)
    )
    A_petsc.assemble()

    # Create PETSc vectors
    b_petsc = PETSc.Vec().createWithArray(b_np)
    x_petsc = PETSc.Vec().createSeq(n)

    # Handle nullspace if requested
    nullvec = None
    nullspace = None
    if remove_nullspace:
        # Create constant nullspace vector
        nullvec = A_petsc.createVecLeft()
        nullvec.set(1.0)
        nullvec.normalize()

        # Create and set nullspace
        nullspace = PETSc.NullSpace().create(vectors=[nullvec])
        A_petsc.setNullSpace(nullspace)

        # Remove nullspace from RHS
        nullspace.remove(b_petsc)

    # Create or reuse KSP solver
    if ksp is None:
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setType(solver_type)
        ksp.setTolerances(rtol=float(tolerance), atol=0, max_it=max_iterations)
        pc = ksp.getPC()
        pc.setType(preconditioner)
        ksp.setFromOptions()
    else:
        # Update operators for existing KSP (reuse preconditioner structure)
        ksp.setOperators(A_petsc)

    # Solve
    ksp.solve(b_petsc, x_petsc)

    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(
            f"PETSc did not converge. Reason: {ksp.getConvergedReason()}, "
            f"Iterations: {ksp.getIterationNumber()}"
        )

    # Convert result to NumPy
    x_np = x_petsc.getArray().copy()

    # Cleanup temporary objects (keep KSP for reuse)
    A_petsc.destroy()
    b_petsc.destroy()
    x_petsc.destroy()
    if nullvec is not None:
        nullvec.destroy()
    if nullspace is not None:
        nullspace.destroy()

    return x_np, ksp
