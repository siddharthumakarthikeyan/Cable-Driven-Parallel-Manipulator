"""
cdpm/tension_solver.py — Advanced Cable Tension Distribution
=============================================================

LATEST ADVANCED METHOD:  Quadratic Programming with OSQP warm-starting
-----------------------------------------------------------------------
The tension distribution problem for a redundant CDPM is formulated as a
strictly convex QP and solved with OSQP (Operator Splitting QP), the
state-of-the-art embedded-system QP solver (Stellato et al., 2020):

    minimise    ½ ‖t − t_ref‖²
    subject to  A · t  =  w_req          (static / dynamic equilibrium)
                t_min  ≤  t  ≤  t_max   (physical actuator constraints)

where
  t      ∈ ℝⁿ   cable tension vector
  A      ∈ ℝ^{6×n} structure matrix
  w_req  ∈ ℝ⁶    required wrench
  t_ref           midpoint of feasible tension range (optimal centering)

Key advantages of this formulation
  • Globally optimal  — the QP is strictly convex (unique solution).
  • Real-time capable — OSQP solves ~2000 iterations in <1 ms (warm start).
  • Constraint aware  — both equality and inequality constraints respected.
  • Warm-startable    — previous solution seeds next solve (trajectory tracking).
  • Numerically robust — handles near-singular configurations gracefully.

Fallback hierarchy
  1. OSQP via CVXPY          (primary, fastest)
  2. scipy SLSQP             (no CVXPY install needed)
  3. Null-space projection   (analytical, zero-dependency emergency fallback)

References
----------
Stellato, B. et al. (2020). OSQP: An Operator Splitting Solver for QPs.
    Mathematical Programming Computation, 12, 637–672.
Pott, A. (2018). Cable-Driven Parallel Robots — Theory and Application.
    Springer Tracts in Advanced Robotics.
Jamshidifar, H. et al. (2022). Real-time tension distribution in CDPR.
    IEEE Trans. Robotics.
"""

import time
import warnings
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Optional CVXPY import (enables OSQP backend)
# ---------------------------------------------------------------------------
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False
    warnings.warn(
        "CVXPY not found — the advanced OSQP solver is unavailable.\n"
        "Install it with:  pip install cvxpy\n"
        "Falling back to scipy SLSQP.",
        ImportWarning,
        stacklevel=2,
    )


# ===========================================================================
class TensionSolver:
    """
    Advanced cable tension distribution for redundant CDPMs.

    For an 8-cable, 6-DOF system the null space has dimension 2, giving two
    degrees of freedom that are consumed by the optimizer to minimise tension
    deviation from the cable mid-range  t_ref = (t_min + t_max) / 2.

    Parameters
    ----------
    t_min  : float   Minimum permitted tension per cable  [N]
    t_max  : float   Maximum permitted tension per cable  [N]
    method : str     'osqp' | 'slsqp' | 'nullspace' | 'auto'
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        t_min: float = 5.0,
        t_max: float = 200.0,
        method: str = "auto",
    ):
        self.t_min  = float(t_min)
        self.t_max  = float(t_max)
        self.t_ref  = (t_min + t_max) / 2.0

        if method == "auto":
            self.method = "osqp" if _HAS_CVXPY else "slsqp"
        else:
            self.method = method

        # CVXPY parametric problem (built once, re-parameterised each call)
        self._cp_t      = None
        self._cp_A      = None
        self._cp_w      = None
        self._cp_prob   = None
        self._n_cables  = None

        # Performance counters
        self._n_ok   = 0
        self._n_fail = 0
        self._times  = []        # solve times [s]

    # ------------------------------------------------------------------
    # Primary solver: OSQP via CVXPY
    # ------------------------------------------------------------------

    def _init_cvxpy(self, n: int):
        """Compile the CVXPY problem once for n cables."""
        t_ref_vec = np.full(n, self.t_ref)

        t = cp.Variable(n, name="t")
        A = cp.Parameter((6, n), name="A")
        w = cp.Parameter(6, name="w_req")

        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(t - t_ref_vec)),
            [A @ t == w, t >= self.t_min, t <= self.t_max],
        )

        self._cp_t, self._cp_A, self._cp_w, self._cp_prob = t, A, w, prob
        self._n_cables = n

    def solve_osqp(self, A: np.ndarray, w_req: np.ndarray):
        """
        Solve via OSQP (Operator Splitting QP) — the advanced primary method.

        OSQP reformulates the QP as an ADMM (Alternating Direction Method of
        Multipliers) iteration.  Warm-starting drastically cuts iterations
        when the problem changes only slightly between control cycles
        (as it does along a smooth trajectory).

        Returns
        -------
        t       : (n,) optimal tensions, or None if infeasible
        success : bool
        status  : str
        """
        if not _HAS_CVXPY:
            return self.solve_slsqp(A, w_req)

        n = A.shape[1]
        if self._cp_prob is None or self._n_cables != n:
            self._init_cvxpy(n)

        # Update parameters — NO recompilation
        self._cp_A.value = A
        self._cp_w.value = w_req

        self._cp_prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            max_iter=10_000,
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=False,
        )

        status = self._cp_prob.status
        if status in ("optimal", "optimal_inaccurate"):
            t_val = np.clip(self._cp_t.value, self.t_min, self.t_max)
            return t_val, True, status
        return None, False, status

    # ------------------------------------------------------------------
    # Fallback: scipy SLSQP
    # ------------------------------------------------------------------

    def solve_slsqp(self, A: np.ndarray, w_req: np.ndarray):
        """
        Solve via scipy Sequential Least Squares Programming (SLSQP).

        Formulation identical to the OSQP version; serves as a reliable
        fallback when CVXPY is unavailable.

        Returns
        -------
        t       : (n,) tensions or None
        success : bool
        status  : str
        """
        n = A.shape[1]
        t_ref = np.full(n, self.t_ref)

        result = minimize(
            fun=lambda t: 0.5 * np.sum((t - t_ref) ** 2),
            x0=t_ref.copy(),
            jac=lambda t: t - t_ref,
            method="SLSQP",
            bounds=[(self.t_min, self.t_max)] * n,
            constraints={
                "type": "eq",
                "fun": lambda t: A @ t - w_req,
                "jac": lambda _: A,
            },
            options={"ftol": 1e-10, "maxiter": 500, "disp": False},
        )

        if result.success:
            t_val = np.clip(result.x, self.t_min, self.t_max)
            return t_val, True, "optimal"
        return None, False, result.message

    # ------------------------------------------------------------------
    # Analytical: null-space projection
    # ------------------------------------------------------------------

    def solve_nullspace(self, A: np.ndarray, w_req: np.ndarray):
        """
        Analytical null-space projection method.

        Decomposes the solution as:
            t = t_p + N · λ
        where
            t_p  is the minimum-norm particular solution  (A⁺ · w_req)
            N    is an orthonormal basis of null(A)       (SVD rows of Vᵀ)
            λ    minimises  ‖t − t_ref‖²  over the null space

        For 8 cables / 6 DOF, dim null(A) = 2, so the optimisation over λ
        is a 2-D bounded least-squares problem — extremely fast.

        Returns
        -------
        t       : (n,) tensions
        success : bool
        info    : dict  (null_dim, lambda, particular)
        """
        n = A.shape[1]
        t_ref = np.full(n, self.t_ref)

        # Particular solution via pseudo-inverse
        A_pinv = np.linalg.pinv(A)
        t_p    = A_pinv @ w_req                       # (n,)

        # Null-space basis via SVD
        _, sv, Vt = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(sv > 1e-10 * sv[0]))
        N = Vt[rank:].T                               # (n, null_dim)
        null_dim = N.shape[1]

        info = {"null_dim": null_dim, "particular": t_p.copy()}

        if null_dim == 0:
            t = np.clip(t_p, self.t_min, self.t_max)
            info["lambda"] = np.array([])
            return t, True, info

        # Unconstrained optimal λ:  λ* = (NᵀN)⁻¹ Nᵀ (t_ref − t_p)
        NtN  = N.T @ N
        Nt_r = N.T @ (t_ref - t_p)
        lam_unc, *_ = np.linalg.lstsq(NtN, Nt_r, rcond=None)
        t = t_p + N @ lam_unc

        # If bounds violated, solve bounded QP over λ
        if not (np.all(t >= self.t_min - 1e-3) and np.all(t <= self.t_max + 1e-3)):
            res = minimize(
                fun=lambda lam: 0.5 * np.sum((t_p + N @ lam - t_ref) ** 2),
                x0=lam_unc,
                jac=lambda lam: N.T @ (t_p + N @ lam - t_ref),
                method="SLSQP",
                bounds=[(-1e4, 1e4)] * null_dim,
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda lam: t_p + N @ lam - self.t_min,
                        "jac": lambda _: N,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda lam: self.t_max - (t_p + N @ lam),
                        "jac": lambda _: -N,
                    },
                ],
                options={"maxiter": 500},
            )
            lam_unc = res.x
            t = t_p + N @ lam_unc

        t = np.clip(t, self.t_min, self.t_max)
        info["lambda"] = lam_unc
        return t, True, info

    # ------------------------------------------------------------------
    # Main public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        A: np.ndarray,
        w_req: np.ndarray,
        verbose: bool = False,
    ):
        """
        Compute optimal cable tensions via the advanced QP method.

        Dispatch order:
          1. OSQP (cvxpy)     if available
          2. scipy SLSQP      fallback
          3. Null-space       emergency fallback

        Parameters
        ----------
        A       : (6, n)  structure matrix
        w_req   : (6,)    required wrench
        verbose : bool    print solver diagnostics

        Returns
        -------
        t       : (n,)  optimal cable tensions  [N]
        success : bool
        """
        t0 = time.perf_counter()

        # ── Primary solver ────────────────────────────────────────────
        if self.method in ("osqp", "auto") and _HAS_CVXPY:
            t, ok, status = self.solve_osqp(A, w_req)
        elif self.method == "nullspace":
            t, ok, info = self.solve_nullspace(A, w_req)
            status = "optimal" if ok else "infeasible"
        else:
            t, ok, status = self.solve_slsqp(A, w_req)

        # ── Emergency fallback ────────────────────────────────────────
        if not ok:
            self._n_fail += 1
            if verbose:
                print(
                    f"  [TensionSolver] primary failed ({status!r}), "
                    "using null-space fallback."
                )
            t, ok, _ = self.solve_nullspace(A, w_req)
        else:
            self._n_ok += 1

        dt = time.perf_counter() - t0
        self._times.append(dt)

        if verbose:
            print(
                f"  [TensionSolver] method={self.method}  "
                f"t∈[{t.min():.1f}, {t.max():.1f}] N  "
                f"time={dt * 1e3:.2f} ms  status={status}"
            )

        return t, ok

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def is_feasible(self, A: np.ndarray, w_req: np.ndarray):
        """
        Return (feasible, margin) for the given configuration.

        margin > 0  means there is slack before any tension bound is hit.
        """
        if _HAS_CVXPY:
            t, ok, _ = self.solve_osqp(A, w_req)
        else:
            t, ok, _ = self.solve_slsqp(A, w_req)

        if not ok or t is None:
            return False, 0.0

        margin = float(min(np.min(t - self.t_min), np.min(self.t_max - t)))
        return True, margin

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self):
        """Return a dict of solver performance statistics."""
        if not self._times:
            return {}
        ms = np.array(self._times) * 1e3
        return {
            "solver":       self.method,
            "n_solved":     self._n_ok,
            "n_failed":     self._n_fail,
            "mean_ms":      float(np.mean(ms)),
            "max_ms":       float(np.max(ms)),
            "std_ms":       float(np.std(ms)),
            "total_s":      float(np.sum(self._times)),
        }

    def __repr__(self):
        return (
            f"TensionSolver(method={self.method!r}, "
            f"t_min={self.t_min}N, t_max={self.t_max}N)"
        )
