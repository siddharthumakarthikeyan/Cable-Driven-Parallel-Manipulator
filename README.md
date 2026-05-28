
# Cable-Driven Parallel Manipulator (CDPM)

A research-grade Python framework for simulation, optimal tension distribution, and cinematic visualization of 8-cable driven parallel manipulators. Designed for advanced robotics research, benchmarking, and reproducible experiments.

## Key Features

- **8-Cable CDPM Modeling:** Modular, extensible robot class with full geometry, kinematics, and structure matrix computation.
- **Trajectory Planning:** Built-in generators for circular, helical, 3D Lissajous, and multi-waypoint quintic trajectories.
- **Optimal Tension Solver:** Advanced QP-based tension distribution (OSQP warm-starting) with fallback to SLSQP and analytical methods.
- **Physics-Based Simulation:** Quasi-static and dynamic-corrected simulation with per-step data logging.
- **3D Visualization & Cinematic Export:** High-quality Matplotlib-based animation, cable tension color mapping, and MP4 export for presentations.
- **Reproducible Research:** Deterministic results, academic license, and citation requirement for scholarly use.

## System Architecture

```
cdpm/
   robot.py           # Robot geometry, kinematics, structure matrix
   tension_solver.py  # QP-based tension distribution (OSQP/SLSQP/analytical)
   simulator.py       # Trajectory simulation engine, data logging
   trajectory.py      # Trajectory generators (circular, helical, Lissajous, quintic)
   visualizer.py      # 3D animation, summary plots, video export
config.py            # Physical and simulation parameters
main.py              # CLI: run, animate, export, benchmark
requirements.txt     # Python dependencies
```

## Technical Highlights

- **QP Tension Distribution:** Formulated as a strictly convex QP, solved via OSQP (warm-started for real-time), with full actuator and cable constraints.
- **Fallback Hierarchy:** Automatic fallback to SLSQP or null-space projection if OSQP/CVXPY unavailable.
- **Trajectory Flexibility:** Easily extendable to new path types; supports smooth, differentiable profiles for advanced control.
- **Visualization:** 3D robot, cable tension colormap, trajectory path, and platform trail; summary plots for cable tensions and platform motion.

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Simulations

```bash
python main.py                   # Default: circular trajectory
python main.py --traj helical    # Helical
python main.py --traj lissajous  # 3D Lissajous
python main.py --traj waypoint   # Multi-waypoint quintic
python main.py --traj all        # Run all, show summary
```

#### Options

- `--no-animate`   Skip animation, show static summary only
- `--save-fig`     Save summary figure as PNG
- `--save-anim`    Save animation as MP4 (requires ffmpeg)
- `--dt 0.02`      Control time step [s]
- `--duration 20`  Simulation duration [s]

## Academic License & Citation

This software is provided under an **Academic License**. Commercial use is strictly prohibited.

If you use this code or its results in your research, **please cite the author's work**:

Siddharth Umakarthikeyan  
[Google Scholar](https://scholar.google.com/citations?user=Rl6gtuoAAAAJ&hl=en&oi=sra)

For citation details, refer to the publications listed on the author's Google Scholar profile.

## Author

[Siddharth Umakarthikeyan](https://www.linkedin.com/in/siddharthind/)
