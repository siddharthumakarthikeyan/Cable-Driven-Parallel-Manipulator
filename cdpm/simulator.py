"""
cdpm/simulator.py — Quasi-static trajectory simulation engine
=============================================================

Drives the CDPM through a desired trajectory by computing, at each time
step, the optimal cable tensions required for equilibrium (or near-
equilibrium under dynamic correction) using the advanced QP solver.

Data recorded per step
----------------------
  time          : simulation time                [s]
  desired_pos   : commanded position              [m]
  actual_pos    : same as desired (quasi-static)  [m]
  cable_lengths : IK-computed lengths             [m]
  tensions      : optimal tensions                [N]
  feasible      : bool — was the QP feasible?
  wrench_error  : ‖A·t − w_req‖₂                 (equilibrium residual)
"""

from __future__ import annotations
import time as _time
import numpy as np
from typing import Callable, Dict, Optional, Tuple

from cdpm.robot import CDPM8Cable
from cdpm.tension_solver import TensionSolver


class SimulationData:
    """Lightweight container for per-step simulation records."""

    def __init__(self, n_cables: int = 8):
        self.time:          list[float]      = []
        self.desired_pos:   list[np.ndarray] = []
        self.cable_lengths: list[np.ndarray] = []
        self.tensions:      list[np.ndarray] = []
        self.feasible:      list[bool]       = []
        self.wrench_error:  list[float]      = []
        self.accel:         list[np.ndarray] = []

    # ── convenience conversions ────────────────────────────────────────
    def as_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "time":          np.array(self.time),
            "desired_pos":   np.array(self.desired_pos),
            "cable_lengths": np.array(self.cable_lengths),
            "tensions":      np.array(self.tensions),
            "feasible":      np.array(self.feasible),
            "wrench_error":  np.array(self.wrench_error),
            "accel":         np.array(self.accel),
        }

    def __len__(self):
        return len(self.time)


# ---------------------------------------------------------------------------

class Simulator:
    """
    Quasi-static (and quasi-dynamic) CDPM simulation.

    At each control cycle the simulator:
      1. Queries the trajectory for the desired position (and acceleration
         for the dynamic correction).
      2. Computes the structure matrix  A  via inverse kinematics.
      3. Assembles the required wrench  w_req  (gravity + inertial term).
      4. Calls the advanced QP tension solver.
      5. Records all data.

    Parameters
    ----------
    robot  : CDPM8Cable instance
    solver : TensionSolver instance
    dt     : control / simulation time step  [s]
    """

    def __init__(
        self,
        robot:  CDPM8Cable,
        solver: TensionSolver,
        dt:     float = 0.02,
        dynamic_correction: bool = True,
    ):
        self.robot   = robot
        self.solver  = solver
        self.dt      = float(dt)
        self.dynamic_correction = dynamic_correction

    # ------------------------------------------------------------------

    def run(
        self,
        trajectory:   Callable,
        duration:     float,
        t_start:      float = 0.0,
        verbose:      bool  = False,
        print_every:  int   = 50,
    ) -> SimulationData:
        """
        Run the full simulation.

        Parameters
        ----------
        trajectory  : callable  t → (pos, vel, acc) each (3,)
        duration    : float     total simulation time  [s]
        t_start     : float     initial time offset    [s]
        verbose     : bool
        print_every : int       print status every N steps

        Returns
        -------
        data : SimulationData
        """
        data   = SimulationData(self.robot.n_cables)
        t_wall = _time.perf_counter()

        n_steps = int(np.ceil(duration / self.dt))
        n_fail  = 0

        for step in range(n_steps + 1):
            t = t_start + step * self.dt

            # ── Desired trajectory ─────────────────────────────────────
            pos, vel, acc = trajectory(t)

            if not self.robot.is_in_workspace(pos):
                if verbose:
                    print(
                        f"  [t={t:.2f}s]  WARNING: position {pos} is outside "
                        "the safe workspace."
                    )

            # ── Structure matrix ───────────────────────────────────────
            try:
                A, lengths, _ = self.robot.structure_matrix(pos)
            except ValueError as exc:
                if verbose:
                    print(f"  [t={t:.2f}s]  IK error: {exc}")
                continue

            # ── Required wrench ────────────────────────────────────────
            a_in = acc if self.dynamic_correction else None
            w_req = self.robot.required_wrench(accel=a_in)

            # ── Tension solve (advanced QP) ────────────────────────────
            tensions, ok = self.solver.solve(A, w_req, verbose=False)
            if not ok:
                n_fail += 1

            # ── Equilibrium residual ───────────────────────────────────
            residual = float(np.linalg.norm(A @ tensions - w_req))

            # ── Record ─────────────────────────────────────────────────
            data.time.append(t)
            data.desired_pos.append(pos.copy())
            data.cable_lengths.append(lengths.copy())
            data.tensions.append(tensions.copy())
            data.feasible.append(ok)
            data.wrench_error.append(residual)
            data.accel.append(acc.copy())

            # ── Console ────────────────────────────────────────────────
            if verbose and step % print_every == 0:
                elapsed = _time.perf_counter() - t_wall
                print(
                    f"  step {step:5d}/{n_steps}  t={t:6.2f}s  "
                    f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}]  "
                    f"t∈[{tensions.min():.1f},{tensions.max():.1f}]N  "
                    f"‖err‖={residual:.2e}  wall={elapsed:.1f}s"
                )

        if verbose:
            print(
                f"\n  Simulation done:  {n_steps+1} steps, "
                f"{n_fail} failed solves, "
                f"wall time={_time.perf_counter()-t_wall:.2f}s"
            )
            stats = self.solver.statistics()
            if stats:
                print(
                    f"  Solver stats: mean={stats['mean_ms']:.2f}ms  "
                    f"max={stats['max_ms']:.2f}ms"
                )

        return data

    # ------------------------------------------------------------------

    def step(
        self,
        position:  np.ndarray,
        accel:     Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Single-step tension computation (for real-time / online use).

        Returns
        -------
        tensions : (n,)
        lengths  : (n,)
        success  : bool
        """
        A, lengths, _ = self.robot.structure_matrix(position)
        w_req = self.robot.required_wrench(accel=accel)
        tensions, ok = self.solver.solve(A, w_req)
        return tensions, lengths, ok
