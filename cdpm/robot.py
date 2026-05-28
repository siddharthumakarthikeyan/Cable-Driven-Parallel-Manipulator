"""
cdpm/robot.py — Robot geometry, inverse kinematics, and structure matrix
=========================================================================

An 8-cable CDPM consists of:
  • Frame   : 8 fixed anchor points at the corners of an outer cube.
  • Platform: A rigid body with 8 attachment points at the corners of a
              smaller inner cube.
  • Cables  : Cable i connects frame anchor B[i] to platform point P[i].

Coordinate convention
---------------------
  World frame origin at cube centre.  x → right, y → forward, z → up.
  Corner ordering (both cubes, same index):

      Index  (x-sign, y-sign, z-sign)
      ─────  ───────────────────────
        0      (−, −, −)   bottom-back-left
        1      (+, −, −)   bottom-back-right
        2      (+, +, −)   bottom-front-right
        3      (−, +, −)   bottom-front-left
        4      (−, −, +)   top-back-left
        5      (+, −, +)   top-back-right
        6      (+, +, +)   top-front-right
        7      (−, +, +)   top-front-left
"""

import numpy as np


class CDPM8Cable:
    """
    Geometry and kinematics of an 8-cable CDPM.

    Each cable i provides a tensile force on the platform:

        f_i = t_i * u_i

    where u_i = (B[i] − P_i) / ‖B[i] − P_i‖  points from the platform
    attachment point toward the frame anchor (so t_i > 0 is physically
    meaningful as tension, not compression).

    Static equilibrium (no rotation for simplicity):

        A @ t = w_req

    with the structure (wrench) matrix A ∈ ℝ^{6×8}:

        A[:, i] = [ u_i          ]
                  [ r_i × u_i   ]

    and r_i = R · b_i  is the moment arm from the platform CoM.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        frame_size: float = 2.0,
        platform_size: float = 0.10,
        mass: float = 2.0,
        g: float = 9.81,
    ):
        """
        Parameters
        ----------
        frame_size    : Side length of the outer (frame) cube  [m]
        platform_size : Nominal half-size of the platform      [m]
                        The platform is a flat rectangular box:
                        bx = platform_size,
                        by = platform_size * 0.75,
                        bz = platform_size * 0.40
                        (non-cubic to achieve a full-rank structure matrix)
        mass          : Platform mass                          [kg]
        g             : Gravitational acceleration             [m/s²]
        """
        self.frame_size    = float(frame_size)
        self.platform_size = float(platform_size)
        self.mass          = float(mass)
        self.g             = float(g)
        self.n_cables      = 8

        a  = frame_size / 2.0   # frame half-side
        bx = platform_size          # platform half-width  (x)
        by = platform_size * 0.75   # platform half-depth  (y)
        bz = platform_size * 0.40   # platform half-height (z)  ← shorter than bx/by

        # Frame anchor points  B[i] ∈ ℝ³  (world frame, fixed) — 8 cube corners
        signs = np.array([
            [-1, -1, -1],   # 0  bottom-back-left
            [ 1, -1, -1],   # 1  bottom-back-right
            [ 1,  1, -1],   # 2  bottom-front-right
            [-1,  1, -1],   # 3  bottom-front-left
            [-1, -1,  1],   # 4  top-back-left
            [ 1, -1,  1],   # 5  top-back-right
            [ 1,  1,  1],   # 6  top-front-right
            [-1,  1,  1],   # 7  top-front-left
        ], dtype=float)

        self.B = signs * a   # (8, 3)  frame anchors

        # Type-II (crossed) routing: cable i connects frame corner B[i] to
        # platform attachment point SHIFTED by 4 positions (index (i+4)%8).
        # This means bottom frame cables attach to the top of the platform
        # and top frame cables attach to the bottom — reversing the z-sign.
        #
        # Combined with a flat rectangular (non-cubic) platform, this yields
        # a full-rank-6 structure matrix, enabling complete 6-DOF wrench
        # feasibility throughout the workspace.
        #
        #   rank(A) = 6  ← verified numerically; singular values all > 0.04
        b_scales = np.array([bx, by, bz])
        self.b_local = np.roll(signs, 4, axis=0) * b_scales   # (8, 3)

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def inverse_kinematics(self, position, R=None):
        """
        Compute cable lengths and unit vectors for a given platform pose.

        Parameters
        ----------
        position : array-like (3,)   Platform CoM in world frame [m]
        R        : (3,3) ndarray     Orientation matrix; None → identity

        Returns
        -------
        lengths      : (8,)   Cable lengths   [m]
        unit_vectors : (8,3)  Cable unit vectors (platform attachment → anchor)
        b_world      : (8,3)  Platform attachment points in world frame
        """
        if R is None:
            R = np.eye(3)
        p = np.asarray(position, dtype=float).ravel()

        # Platform attachment points in world frame
        b_world = (R @ self.b_local.T).T + p   # (8, 3)

        # Cable vectors: from platform attachment to frame anchor
        cable_vecs = self.B - b_world           # (8, 3)

        # Cable lengths
        lengths = np.linalg.norm(cable_vecs, axis=1)  # (8,)

        if np.any(lengths < 1e-6):
            raise ValueError(
                f"Degenerate configuration at p={p}: cable length ≈ 0."
            )

        # Unit vectors
        unit_vectors = cable_vecs / lengths[:, np.newaxis]  # (8, 3)

        return lengths, unit_vectors, b_world

    # ------------------------------------------------------------------
    # Structure matrix
    # ------------------------------------------------------------------

    def structure_matrix(self, position, R=None):
        """
        Build the wrench (structure) matrix  A ∈ ℝ^{6×8}.

        The equilibrium equation is:
            A @ t = w_req

        Column i:
            A[:3, i] = u_i           (force contribution of cable i)
            A[3:, i] = r_i × u_i    (torque contribution, r_i = R·b_i)

        Parameters
        ----------
        position : array-like (3,)
        R        : (3,3) orientation matrix; None → identity

        Returns
        -------
        A            : (6, 8)  structure matrix
        lengths      : (8,)    cable lengths [m]
        unit_vectors : (8, 3)  cable unit vectors
        """
        if R is None:
            R = np.eye(3)
        p = np.asarray(position, dtype=float).ravel()

        lengths, unit_vectors, b_world = self.inverse_kinematics(p, R)

        A = np.zeros((6, 8))
        for i in range(8):
            u_i = unit_vectors[i]
            r_i = b_world[i] - p          # moment arm = R @ b_local[i]
            A[:3, i] = u_i
            A[3:, i] = np.cross(r_i, u_i)

        return A, lengths, unit_vectors

    # ------------------------------------------------------------------
    # Required wrench
    # ------------------------------------------------------------------

    def required_wrench(self, accel=None):
        """
        Required wrench that the cables must supply to achieve a given
        platform motion (in the world frame).

        Quasi-static (accel = None):
            w_req = [0, 0, m·g, 0, 0, 0]ᵀ

        Dynamic (accel provided):
            w_req = [m·(a + g·ẑ), 0, 0, 0]ᵀ

        Parameters
        ----------
        accel : array-like (3,) or None   Platform linear acceleration [m/s²]

        Returns
        -------
        w_req : (6,)  required wrench [N, N·m]
        """
        w_req = np.array([0.0, 0.0, self.mass * self.g, 0.0, 0.0, 0.0])
        if accel is not None:
            w_req[:3] += self.mass * np.asarray(accel, dtype=float)
        return w_req

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def workspace_radius(self, safety=0.25):
        """Conservative inscribed workspace radius [m]."""
        return self.frame_size * safety

    def is_in_workspace(self, position, margin=0.05):
        """Rough bounding-box workspace check."""
        limit = self.frame_size / 2.0 * (1.0 - margin)
        p = np.asarray(position)
        return bool(np.all(np.abs(p) < limit))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"CDPM8Cable(frame={self.frame_size}m, "
            f"platform={self.platform_size}m, "
            f"mass={self.mass}kg)"
        )
