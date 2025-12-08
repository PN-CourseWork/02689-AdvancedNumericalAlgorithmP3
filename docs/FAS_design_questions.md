# FAS Implementation - Design Questions

Please answer each question by writing your response below the question.

---
from the paper: which represents the prolongation of correction from coarse grid to
ﬁne grid with Pkk1 the prolongation operator. In this FAS scheme,
the coefﬁcient matrix and the source term on the coarse grid are up-
dated using the newly-computed variables and the variables are
calculated on every grid level, this feature preserves the nonlinear
property of the discretized equation and helps to accelerate the
convergence.

## Q1: Coarsest Grid Treatment

The paper notation "48-VMG-111" means 1 pre-smoothing iteration on each of the 3 levels (finest, middle, coarsest).

**Options:**
- **(A)** Same iterations on coarsest as other levels (paper approach) YES
- **(B)** Solve coarsest grid to full convergence (more expensive but potentially more accurate) NO 
- **(C)** Use more iterations on coarsest (e.g., 3-5 steps) but don't fully converge PERHAPS

**Your answer:**


---

## Q2: Corner Singularity Treatment on Coarse Grids

The subtraction method (Botella & Peyret) computes singular velocity derivatives that blow up near corners. On very coarse grids (N<8), this causes numerical overflow.

Current VMG uses a hybrid approach:
- N ≥ 8: Use subtraction method (matches fine grid)
- N < 8: Fall back to smoothing method

**Options:**
- **(A)** Keep hybrid approach (subtraction on fine, smoothing on very coarse)
- **(B)** Always use smoothing method on all levels (simpler, slightly less accurate) Always go with this! But make general so later on we can use regularized lid driven cavity! 
- **(C)** Always use subtraction but with corner exclusion mask (current VMG approach)
- **(D)** Match whatever the finest level uses

**Your answer:**


---

## Q3: Code Reuse vs Fresh Implementation

**Option A - Reuse existing code:**
- Reuse `SpectralLevel` dataclass from `multigrid/fsg.py`
- Reuse `MultigridSmoother` class for RK4 stepping
- Reuse transfer operators from `operators/transfer_operators.py`
- Just write new V-cycle logic in `fas.py`

**Option B - Fresh implementation:**
- Create new `FASLevel` dataclass (simpler, only what's needed)
- Inline RK4 stepping in the solver
- Still reuse transfer operators
- Cleaner but some code duplication
Go with the cleanest approach. But maybe start over if you happen to find mistakes somewhere...

**Your answer:**


---

## Q4: CFL Number Adaptation

Paper notes:
- CFL = 2.5 works for large N (stable)
- CFL = 0.3-0.5 needed for small N (N=4-10) to converge

Current config uses fixed CFL = 2.0.

**Options:**
- **(A)** Keep fixed CFL = 2.0 for all levels Keep fixed! 
- **(B)** Auto-adapt CFL based on N: `CFL = min(2.5, 0.3 + 0.05*N)`
- **(C)** Use different CFL per level (config parameter)
- **(D)** Other (specify)

**Your answer:**


---

## Q5: Convergence Logging

How verbose should the convergence output be?

**Options:**
- **(A)** Minimal: Only log when converged or failed
- **(B)** Moderate: Log every 10-50 V-cycles with E_RMS value
- **(C)** Verbose: Log every V-cycle Verbose so we can debug!
- **(D)** Configurable via log level

**Your answer:**


---

## Q6: Additional Features

Should the new FAS solver support any of these optional features?

- [ ] FMG (Full Multigrid) as initialization strategy No 
- [ ] W-cycle option (in addition to V-cycle) No 
- [ ] Adaptive number of pre-smoothing steps based on residual reduction No 
- [ ] Save intermediate solutions for visualization

**Check the ones you want, or write "none" / "later":**

we should be able to specify the number of levels though!


---

## Q7: Default Tolerance

Paper uses:
- ε = 10⁻⁴ for lid-driven cavity (practical)
- ε = 10⁻¹⁰ for test problems (validation)

Current config uses tolerance = 1e-6.

**What should the default be for FAS?**

**Your answer:**


---

## Q8: Naming Convention

What should we call this solver?

**Options:**
- **(A)** `FASSolver` / `fas.py` / `fas.yaml` FAS.py!
- **(B)** `FASVCycleSolver` / `fas_vcycle.py` / `fas_vcycle.yaml`
- **(C)** `VMG2Solver` / `vmg2.py` / `vmg2.yaml` (improved VMG)


additionally: You should really verify each component in isolation before doing end-to-end testing! Your tests should not run for longer than 10-20 seconds!
- **(D)** Other (specify)

**Your answer:**


---

## Summary of Your Choices

Once you've answered above, I'll summarize here:

| Question | Your Choice |
|----------|-------------|
| Q1: Coarsest grid | |
| Q2: Corner treatment | |
| Q3: Code reuse | |
| Q4: CFL adaptation | |
| Q5: Logging | |
| Q6: Extra features | |
| Q7: Tolerance | |
| Q8: Naming | |

---

*Please edit this file with your answers and let me know when you're done!*


additionally: You should really verify each component in isolation before doing end-to-end testing! Your tests should not run for longer than 10-20 seconds!
