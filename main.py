"""
main.py — 8-Cable CDPM Simulation with Advanced QP Tension Distribution
========================================================================

Demonstrates four trajectory types for an 8-cable CDPM attached to the
8 corners of a 2 m³ cube.  Cable tensions are resolved at every control
cycle using the **advanced QP method (OSQP warm-starting)**, which
guarantees globally optimal tension distribution while satisfying both
cable-taut (t ≥ t_min) and actuator-limit (t ≤ t_max) constraints.

Run
---
    python main.py                   # circular trajectory (default)
    python main.py --traj helical    # helical
    python main.py --traj lissajous  # 3-D Lissajous
    python main.py --traj waypoint   # multi-waypoint quintic
    python main.py --traj all        # run all four & show summary

Options
-------
    --no-animate     skip animation, show static summary only
    --save-fig       save summary figure as PNG
    --save-anim      save animation as MP4 (requires ffmpeg)
    --dt 0.02        control time step [s]
    --duration 20    simulation duration [s]
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from config import (
    FRAME_SIZE, PLATFORM_SIZE, PLATFORM_MASS, GRAVITY,
    T_MIN, T_MAX, DT, SIM_DURATION,
    TRAJ_RADIUS, TRAJ_FREQ, TRAJ_CENTER,
)
from cdpm.robot       import CDPM8Cable
from cdpm.tension_solver import TensionSolver
from cdpm.trajectory  import TrajectoryPlanner
from cdpm.simulator   import Simulator
from cdpm.visualizer  import Visualizer


# ===========================================================================
# Build system
# ===========================================================================

def build_system(dt: float = DT):
    robot  = CDPM8Cable(
        frame_size=FRAME_SIZE,
        platform_size=PLATFORM_SIZE,
        mass=PLATFORM_MASS,
        g=GRAVITY,
    )
    solver = TensionSolver(t_min=T_MIN, t_max=T_MAX, method="auto")
    planner = TrajectoryPlanner(frame_size=FRAME_SIZE)
    sim    = Simulator(robot, solver, dt=dt, dynamic_correction=True)
    return robot, solver, planner, sim


# ===========================================================================
# Trajectory definitions
# ===========================================================================

def make_circular(planner: TrajectoryPlanner):
    print("  Trajectory: circular  (XY plane)")
    return planner.circular(
        centre=TRAJ_CENTER,
        radius=TRAJ_RADIUS,
        freq_hz=TRAJ_FREQ,
        plane="xy",
    )


def make_helical(planner: TrajectoryPlanner, duration: float):
    print("  Trajectory: helical  (rising helix)")
    n_revs  = TRAJ_FREQ * duration
    z_rise  = min(0.20, FRAME_SIZE * 0.15)   # total rise capped to 15% of frame
    pitch   = z_rise / max(n_revs, 0.5)
    return planner.helical(
        centre=[0.0, 0.0, -z_rise / 2.0],
        radius=TRAJ_RADIUS * 0.85,
        freq_hz=TRAJ_FREQ,
        pitch=pitch,
    )


def make_lissajous(planner: TrajectoryPlanner):
    print("  Trajectory: Lissajous  (3-D figure)")
    return planner.lissajous(
        centre=TRAJ_CENTER,
        amplitude=[TRAJ_RADIUS, TRAJ_RADIUS, TRAJ_RADIUS * 0.5],
        freq_ratio=[1.0, 2.0, 3.0],
        base_freq=TRAJ_FREQ * 0.5,
        phase=[0.0, np.pi / 4, np.pi / 2],
    )


def make_waypoint(planner: TrajectoryPlanner, duration: float):
    print("  Trajectory: multi-waypoint quintic  (5 waypoints)")
    r = TRAJ_RADIUS
    wpts = [
        [ 0.00,  0.00,  0.00],
        [ r,     0.00,  0.10],
        [ 0.00,  r,    -0.10],
        [-r,     0.00,  0.10],
        [ 0.00, -r,    -0.10],
        [ 0.00,  0.00,  0.00],
    ]
    return planner.multi_waypoint(
        waypoints=[np.array(w) for w in wpts],
        total_duration=duration,
    )


_TRAJ_MAP = {
    "circular":  make_circular,
    "helical":   make_helical,
    "lissajous": make_lissajous,
    "waypoint":  make_waypoint,
}


# ===========================================================================
# Run one scenario
# ===========================================================================

def run_scenario(
    name:       str,
    planner:    TrajectoryPlanner,
    sim:        Simulator,
    duration:   float,
    verbose:    bool = True,
):
    print(f"\n{'='*60}")
    print(f"  Scenario: {name.upper()}")
    print(f"{'='*60}")

    builder = _TRAJ_MAP[name]
    # Some builders need duration
    if name in ("helical", "waypoint"):
        traj = builder(planner, duration)
    else:
        traj = builder(planner)

    data = sim.run(traj, duration=duration, verbose=verbose, print_every=50)

    arr = data.as_arrays()
    T   = arr["tensions"]
    print(f"\n  Tension statistics:")
    print(f"    Mean  per cable : {T.mean(axis=0).round(1)} N")
    print(f"    Min   overall   : {T.min():.1f} N")
    print(f"    Max   overall   : {T.max():.1f} N")
    print(f"    Feasible steps  : {arr['feasible'].sum()}/{len(arr['feasible'])}")

    stats = sim.solver.statistics()
    if stats:
        print(f"\n  Solver: {stats['solver'].upper()}")
        print(f"    Mean solve time : {stats['mean_ms']:.3f} ms")
        print(f"    Max  solve time : {stats['max_ms']:.3f} ms")
        print(f"    Successful      : {stats['n_solved']}")
        print(f"    Failed          : {stats['n_failed']}")

    return data


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="8-Cable CDPM Simulation with Advanced QP Tension Distribution"
    )
    parser.add_argument(
        "--traj",
        choices=["circular", "helical", "lissajous", "waypoint", "all"],
        default="circular",
        help="Trajectory type to simulate (default: circular)",
    )
    parser.add_argument("--no-animate", action="store_true",
                        help="Skip animation; show static plots only")
    parser.add_argument("--save-fig",  action="store_true",
                        help="Save summary figure as PNG")
    parser.add_argument("--save-anim", action="store_true",
                        help="Save animation as MP4 (requires ffmpeg)")
    parser.add_argument("--dt",       type=float, default=DT,
                        help=f"Time step [s] (default {DT})")
    parser.add_argument("--duration", type=float, default=SIM_DURATION,
                        help=f"Simulation duration [s] (default {SIM_DURATION})")
    args = parser.parse_args()

    # ── Banner ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  8-CABLE DRIVEN PARALLEL MANIPULATOR (CDPM)")
    print("  Advanced QP Tension Distribution (OSQP warm-start)")
    print("═"*60)
    print(f"  Frame size    : {FRAME_SIZE} m³ cube")
    print(f"  Platform mass : {PLATFORM_MASS} kg")
    print(f"  Tension range : [{T_MIN}, {T_MAX}] N")
    print(f"  Time step     : {args.dt*1000:.0f} ms  ({1/args.dt:.0f} Hz)")
    print(f"  Duration      : {args.duration} s")

    robot, solver, planner, sim = build_system(dt=args.dt)

    # ── Run scenario(s) ────────────────────────────────────────────────
    names = (
        ["circular", "helical", "lissajous", "waypoint"]
        if args.traj == "all"
        else [args.traj]
    )

    for name in names:
        # Reset solver warm-start state between scenarios
        robot2, solver2, planner2, sim2 = build_system(dt=args.dt)
        data = run_scenario(name, planner2, sim2, args.duration)

        viz  = Visualizer(robot2, data,
                          title=f"8-Cable CDPM — {name.capitalize()} Trajectory")

        save_fig  = f"results_{name}.png"  if args.save_fig  else None
        save_anim = f"anim_{name}.mp4"     if args.save_anim else None

        if not args.no_animate:
            print("\n  Starting animation …  (close window to continue)")
            viz.animate(interval_ms=int(args.dt * 1000),
                        save_path=save_anim,
                        show=True)

        print("\n  Showing static summary …")
        viz.plot_summary(save_path=save_fig, show=True)

    print("\n  Done.")


if __name__ == "__main__":
    main()
