# =============================================================================
# 8-Cable Driven Parallel Manipulator — Configuration
# =============================================================================

# ── Robot Geometry ────────────────────────────────────────────────────────────
FRAME_SIZE      = 2.0     # Side length of outer (frame) cube  [m]
PLATFORM_SIZE   = 0.10    # Platform nominal half-size [m] → box 0.20×0.15×0.08 m

# ── Physical Parameters ───────────────────────────────────────────────────────
PLATFORM_MASS   = 2.0     # Platform mass                      [kg]
GRAVITY         = 9.81    # Gravitational acceleration         [m/s²]

# ── Cable Parameters ──────────────────────────────────────────────────────────
T_MIN           = 5.0     # Minimum cable tension (must be > 0 to stay taut) [N]
T_MAX           = 200.0   # Maximum cable tension (actuator limit)            [N]

# ── Simulation ────────────────────────────────────────────────────────────────
DT              = 0.02    # Time step  [s]  (50 Hz)
SIM_DURATION    = 20.0    # Total simulation time              [s]

# ── Trajectory ────────────────────────────────────────────────────────────────
# All trajectories are centred at the workspace origin.
TRAJ_RADIUS     = 0.30    # Radius of circular / helical path  [m]
TRAJ_FREQ       = 0.15    # Traversal frequency                [Hz]
TRAJ_CENTER     = [0.0, 0.0, 0.0]   # Centre of trajectory   [m]

# ── Visualisation ─────────────────────────────────────────────────────────────
ANIMATE_SPEED   = 1.0     # Real-time speed multiplier
SAVE_ANIMATION  = False   # Write animation to MP4
FIG_DPI         = 100
