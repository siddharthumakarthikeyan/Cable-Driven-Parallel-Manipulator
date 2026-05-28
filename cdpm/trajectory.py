"""
cdpm/trajectory.py — Trajectory generators for CDPM motion planning
=====================================================================

Provides smooth, differentiable position (and optionally velocity /
acceleration) profiles for trajectory-tracking control:

  • QuinticPTP      — point-to-point via 5th-order polynomial (zero
                      velocity and acceleration at start / end).
  • CircularPath    — constant-speed circle in any axis-aligned plane.
  • HelicalPath     — ascending helix (circle + linear z-rise).
  • LissajousPath   — 3-D Lissajous figure.
  • MultiWaypointPTP— chain of quintic segments through N waypoints.
  • TrajectoryPlanner — convenience factory that wires everything up.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _quintic_scalar(q0: float, qf: float, T: float, t: float):
    """
    5th-order (quintic) polynomial scalar trajectory.

    Boundary conditions:
        q(0)  = q0,   q(T)  = qf
        q'(0) = 0,    q'(T) = 0
        q''(0)= 0,    q''(T)= 0

    Returns (q, dq, ddq).
    """
    tau = float(np.clip(t / T, 0.0, 1.0))
    tau2, tau3, tau4, tau5 = tau**2, tau**3, tau**4, tau**5

    # Normalised position polynomial
    s   = 10*tau3 - 15*tau4 +  6*tau5
    ds  = (30*tau2 - 60*tau3 + 30*tau4) / T
    dds = (60*tau - 180*tau2 + 120*tau3) / (T**2)

    q   = q0 + (qf - q0) * s
    dq  = (qf - q0) * ds
    ddq = (qf - q0) * dds
    return q, dq, ddq


# ---------------------------------------------------------------------------
# Individual trajectory classes
# ---------------------------------------------------------------------------

class QuinticPTP:
    """
    Single-segment point-to-point trajectory using a quintic polynomial.
    Guarantees continuous position, velocity, and acceleration with
    zero boundary velocities and accelerations.
    """

    def __init__(
        self,
        start: np.ndarray,
        end:   np.ndarray,
        duration: float = 5.0,
        t_start: float = 0.0,
    ):
        self.start    = np.asarray(start, dtype=float)
        self.end      = np.asarray(end,   dtype=float)
        self.duration = float(duration)
        self.t_start  = float(t_start)
        self.t_end    = t_start + duration

    def __call__(self, t: float):
        tau = t - self.t_start
        pos = np.empty(3)
        vel = np.empty(3)
        acc = np.empty(3)
        for k in range(3):
            pos[k], vel[k], acc[k] = _quintic_scalar(
                self.start[k], self.end[k], self.duration, tau
            )
        return pos, vel, acc


class CircularPath:
    """
    Constant-speed circular trajectory.

    Parameters
    ----------
    centre   : (3,)   centre of the circle
    radius   : float  radius  [m]
    omega    : float  angular velocity  [rad/s]   (positive = CCW when viewed
                      from +z for 'xy' plane)
    plane    : str    'xy' | 'xz' | 'yz'
    phi0     : float  initial phase offset  [rad]
    """

    _PLANES = {
        "xy": (0, 1, 2),
        "xz": (0, 2, 1),
        "yz": (1, 2, 0),
    }

    def __init__(
        self,
        centre: np.ndarray,
        radius: float = 0.30,
        omega:  float = 2 * np.pi * 0.1,
        plane:  str   = "xy",
        phi0:   float = 0.0,
    ):
        self.centre = np.asarray(centre, dtype=float)
        self.radius = float(radius)
        self.omega  = float(omega)
        self.plane  = plane
        self.phi0   = float(phi0)

        if plane not in self._PLANES:
            raise ValueError(f"plane must be one of {list(self._PLANES)}")

    def __call__(self, t: float):
        i0, i1, i2 = self._PLANES[self.plane]
        phi  = self.omega * t + self.phi0
        pos  = self.centre.copy()
        pos[i0] += self.radius * np.cos(phi)
        pos[i1] += self.radius * np.sin(phi)

        vel  = np.zeros(3)
        vel[i0] = -self.radius * self.omega * np.sin(phi)
        vel[i1] =  self.radius * self.omega * np.cos(phi)

        acc  = np.zeros(3)
        acc[i0] = -self.radius * self.omega**2 * np.cos(phi)
        acc[i1] = -self.radius * self.omega**2 * np.sin(phi)

        return pos, vel, acc


class HelicalPath:
    """
    Helical trajectory: circle in XY with linear Z-rise.

    Parameters
    ----------
    centre   : (3,)  centre at  z = centre[2]
    radius   : float
    omega    : float  angular velocity  [rad/s]
    pitch    : float  vertical rise per full revolution  [m/rev]
    """

    def __init__(
        self,
        centre: np.ndarray,
        radius: float = 0.25,
        omega:  float = 2 * np.pi * 0.10,
        pitch:  float = 0.20,
        phi0:   float = 0.0,
    ):
        self.centre = np.asarray(centre, dtype=float)
        self.radius = float(radius)
        self.omega  = float(omega)
        self.pitch  = float(pitch)
        self.phi0   = float(phi0)
        # vertical rate  dz/dt = pitch * omega / (2π)
        self._zdot  = pitch * omega / (2.0 * np.pi)

    def __call__(self, t: float):
        phi = self.omega * t + self.phi0
        pos = self.centre.copy()
        pos[0] += self.radius * np.cos(phi)
        pos[1] += self.radius * np.sin(phi)
        pos[2] += self._zdot * t

        vel = np.array([
            -self.radius * self.omega * np.sin(phi),
             self.radius * self.omega * np.cos(phi),
             self._zdot,
        ])

        acc = np.array([
            -self.radius * self.omega**2 * np.cos(phi),
            -self.radius * self.omega**2 * np.sin(phi),
             0.0,
        ])
        return pos, vel, acc


class LissajousPath:
    """
    3-D Lissajous trajectory.

    pos[k] = centre[k] + amplitude[k] * sin(freq[k] * t + phase[k])
    """

    def __init__(
        self,
        centre:    np.ndarray,
        amplitude: np.ndarray,
        freq:      np.ndarray,
        phase:     Optional[np.ndarray] = None,
    ):
        self.centre    = np.asarray(centre,    dtype=float)
        self.amplitude = np.asarray(amplitude, dtype=float)
        self.freq      = np.asarray(freq,      dtype=float)
        self.phase     = np.zeros(3) if phase is None else np.asarray(phase, dtype=float)

    def __call__(self, t: float):
        phi = self.freq * t + self.phase
        pos = self.centre + self.amplitude * np.sin(phi)
        vel = self.amplitude * self.freq * np.cos(phi)
        acc = -self.amplitude * self.freq**2 * np.sin(phi)
        return pos, vel, acc


class MultiWaypointPTP:
    """
    Chain of quintic PTP segments through an ordered list of waypoints.

    Each segment has equal duration = total_duration / (n_waypoints − 1).
    Velocities / accelerations at interior waypoints are zero (conservative
    but easy to guarantee cable feasibility).
    """

    def __init__(
        self,
        waypoints: List[np.ndarray],
        total_duration: float = 20.0,
    ):
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints.")

        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.n_seg = len(waypoints) - 1
        self.seg_dur = total_duration / self.n_seg
        self.total_duration = total_duration

        self._segments = []
        for i in range(self.n_seg):
            self._segments.append(
                QuinticPTP(
                    start=waypoints[i],
                    end=waypoints[i + 1],
                    duration=self.seg_dur,
                    t_start=i * self.seg_dur,
                )
            )

    def __call__(self, t: float):
        t = float(np.clip(t, 0.0, self.total_duration))
        # Find active segment
        idx = min(int(t / self.seg_dur), self.n_seg - 1)
        return self._segments[idx](t)


# ---------------------------------------------------------------------------
# High-level planner
# ---------------------------------------------------------------------------

class TrajectoryPlanner:
    """
    Convenience factory for creating and switching trajectory profiles.

    Usage
    -----
    >>> planner = TrajectoryPlanner(frame_size=2.0)
    >>> traj = planner.circular(radius=0.3, freq_hz=0.1)
    >>> pos, vel, acc = traj(t=2.5)
    """

    def __init__(self, frame_size: float = 2.0, safety: float = 0.20):
        self._max_r = frame_size * safety    # conservative workspace radius

    def _check_radius(self, radius: float):
        if radius > self._max_r:
            import warnings
            warnings.warn(
                f"Requested radius {radius:.3f} m exceeds conservative "
                f"workspace radius {self._max_r:.3f} m.  "
                "Cable feasibility is not guaranteed.",
                RuntimeWarning,
            )

    def circular(
        self,
        centre:   np.ndarray | list = (0, 0, 0),
        radius:   float = 0.30,
        freq_hz:  float = 0.10,
        plane:    str   = "xy",
        phi0:     float = 0.0,
    ) -> CircularPath:
        self._check_radius(radius)
        return CircularPath(
            centre=np.asarray(centre, dtype=float),
            radius=radius,
            omega=2.0 * np.pi * freq_hz,
            plane=plane,
            phi0=phi0,
        )

    def helical(
        self,
        centre:  np.ndarray | list = (0, 0, -0.15),
        radius:  float = 0.25,
        freq_hz: float = 0.10,
        pitch:   float = 0.30,
        phi0:    float = 0.0,
    ) -> HelicalPath:
        self._check_radius(radius)
        return HelicalPath(
            centre=np.asarray(centre, dtype=float),
            radius=radius,
            omega=2.0 * np.pi * freq_hz,
            pitch=pitch,
            phi0=phi0,
        )

    def lissajous(
        self,
        centre:    np.ndarray | list = (0, 0, 0),
        amplitude: np.ndarray | list = (0.25, 0.25, 0.15),
        freq_ratio: Tuple[float, float, float] = (1.0, 2.0, 3.0),
        base_freq:  float = 0.10,
        phase:      np.ndarray | list = (0.0, np.pi / 4, np.pi / 2),
    ) -> LissajousPath:
        amp = np.asarray(amplitude, dtype=float)
        self._check_radius(float(np.max(amp)))
        freq = np.asarray(freq_ratio) * 2.0 * np.pi * base_freq
        return LissajousPath(
            centre=np.asarray(centre, dtype=float),
            amplitude=amp,
            freq=freq,
            phase=np.asarray(phase, dtype=float),
        )

    def point_to_point(
        self,
        start:    np.ndarray | list,
        end:      np.ndarray | list,
        duration: float = 5.0,
    ) -> QuinticPTP:
        return QuinticPTP(
            start=np.asarray(start, dtype=float),
            end=np.asarray(end,   dtype=float),
            duration=duration,
        )

    def multi_waypoint(
        self,
        waypoints:       List,
        total_duration:  float = 20.0,
    ) -> MultiWaypointPTP:
        return MultiWaypointPTP(waypoints, total_duration)
