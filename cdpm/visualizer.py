"""
cdpm/visualizer.py — 3-D animated visualisation of the CDPM
============================================================

Layout
------
  Left  (3-D axes): robot in space
      • Outer cube frame (grey wireframe)
      • Moving platform (blue filled box)
      • 8 cables coloured by normalised tension (blue→red colormap)
      • Desired trajectory path (dashed)
      • Platform CoM trail (green line)

  Right (2 subplots):
      • Cable tensions vs time (8 coloured lines)
      • Platform XYZ position vs time

Usage
-----
    from cdpm.visualizer import Visualizer
    viz = Visualizer(robot, data)
    viz.animate()          # live window
    viz.plot_summary()     # static overview figure
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from typing import Optional

from cdpm.robot import CDPM8Cable
from cdpm.simulator import SimulationData


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_CABLE_CMAP   = cm.get_cmap("plasma")
_CABLE_COLORS = [_CABLE_CMAP(i / 7) for i in range(8)]


def _cube_edges(half: float):
    """Return a list of (start, end) corner pairs for a cube wireframe."""
    c = half
    corners = np.array([
        [-c, -c, -c], [ c, -c, -c], [ c,  c, -c], [-c,  c, -c],
        [-c, -c,  c], [ c, -c,  c], [ c,  c,  c], [-c,  c,  c],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # bottom face
        (4,5),(5,6),(6,7),(7,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # verticals
    ]
    return [(corners[a], corners[b]) for a, b in edges]


def _platform_lines(position: np.ndarray, b_world: np.ndarray):
    """Lines for the 8-corner platform box."""
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    segs = []
    for a, b in edges:
        segs.append([b_world[a], b_world[b]])
    return segs


# ---------------------------------------------------------------------------

class Visualizer:
    """
    Interactive / animated visualiser for an 8-cable CDPM simulation.

    Parameters
    ----------
    robot : CDPM8Cable
    data  : SimulationData (post-simulation result)
    title : str   window title
    """

    def __init__(
        self,
        robot:  CDPM8Cable,
        data:   SimulationData,
        title:  str = "8-Cable CDPM — Advanced QP Tension Distribution",
    ):
        self.robot = robot
        self.data  = data
        self.title = title
        self._arr  = data.as_arrays()

    # ------------------------------------------------------------------
    # Static summary figure
    # ------------------------------------------------------------------

    def plot_summary(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot a 4-panel static summary of the simulation:
          (1) 3-D trajectory
          (2) Cable tensions vs time
          (3) Platform XYZ vs time
          (4) Tension distribution (box-whisker)
        """
        arr = self._arr
        t   = arr["time"]
        T   = arr["tensions"]           # (N, 8)
        pos = arr["desired_pos"]        # (N, 3)

        fig = plt.figure(figsize=(16, 10), dpi=100)
        fig.suptitle(self.title, fontsize=13, fontweight="bold")
        gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35)

        # ── (1) 3-D trajectory ──────────────────────────────────────
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        self._draw_frame(ax3d)
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                  "g-", lw=1.5, label="Trajectory")
        ax3d.scatter(*pos[0],  s=60, c="lime",   zorder=5, label="Start")
        ax3d.scatter(*pos[-1], s=60, c="orange", zorder=5, label="End")
        ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
        ax3d.set_title("3-D Workspace")
        ax3d.legend(fontsize=8)

        # ── (2) Cable tensions ──────────────────────────────────────
        ax_t = fig.add_subplot(gs[0, 1:])
        for i in range(8):
            ax_t.plot(t, T[:, i], color=_CABLE_COLORS[i],
                      lw=1.2, label=f"Cable {i+1}", alpha=0.9)
        ax_t.axhline(5.0,   color="steelblue", ls="--", lw=0.9, alpha=0.6, label="t_min")
        ax_t.axhline(200.0, color="firebrick", ls="--", lw=0.9, alpha=0.6, label="t_max")
        ax_t.set_xlabel("Time [s]")
        ax_t.set_ylabel("Tension [N]")
        ax_t.set_title("Cable Tensions  (Advanced QP Distribution)")
        ax_t.legend(ncol=4, fontsize=7, loc="upper right")
        ax_t.grid(True, alpha=0.3)

        # ── (3) Platform position ────────────────────────────────────
        ax_p = fig.add_subplot(gs[1, 1])
        labels = ["X", "Y", "Z"]
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for k, (lbl, col) in enumerate(zip(labels, colors)):
            ax_p.plot(t, pos[:, k], color=col, lw=1.4, label=lbl)
        ax_p.set_xlabel("Time [s]")
        ax_p.set_ylabel("Position [m]")
        ax_p.set_title("Platform Position")
        ax_p.legend(fontsize=9)
        ax_p.grid(True, alpha=0.3)

        # ── (4) Tension box-whisker ─────────────────────────────────
        ax_b = fig.add_subplot(gs[1, 2])
        bp   = ax_b.boxplot(
            [T[:, i] for i in range(8)],
            labels=[f"C{i+1}" for i in range(8)],
            patch_artist=True,
            medianprops={"color": "red", "lw": 2},
        )
        for patch, col in zip(bp["boxes"], _CABLE_COLORS):
            patch.set_facecolor(mcolors.to_rgba(col, alpha=0.6))
        ax_b.set_xlabel("Cable")
        ax_b.set_ylabel("Tension [N]")
        ax_b.set_title("Tension Distribution")
        ax_b.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"  Figure saved to {save_path}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Animated visualisation
    # ------------------------------------------------------------------

    def animate(
        self,
        interval_ms: int  = 40,
        trail_len:   int  = 80,
        save_path:   Optional[str] = None,
        show:        bool = True,
    ):
        """
        Animate the CDPM motion with cable tension coloring.

        Parameters
        ----------
        interval_ms : int   delay between frames [ms]
        trail_len   : int   number of trailing positions to show
        save_path   : str   if given, save animation as MP4
        show        : bool  call plt.show() after setup
        """
        arr      = self._arr
        pos_data = arr["desired_pos"]   # (N, 3)
        T_data   = arr["tensions"]      # (N, 8)
        t_data   = arr["time"]
        N        = len(t_data)

        t_all_min = T_data.min()
        t_all_max = T_data.max()
        norm      = mcolors.Normalize(vmin=t_all_min, vmax=t_all_max)
        cmap      = cm.get_cmap("plasma")

        # ── Figure layout ───────────────────────────────────────────
        fig = plt.figure(figsize=(16, 8), dpi=90)
        fig.suptitle(self.title, fontsize=12, fontweight="bold")

        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax_t = fig.add_subplot(2, 2, 2)
        ax_p = fig.add_subplot(2, 2, 4)

        # ── Axes limits ─────────────────────────────────────────────
        hl = self.robot.frame_size / 2.0 * 1.05
        ax3d.set_xlim(-hl, hl); ax3d.set_ylim(-hl, hl); ax3d.set_zlim(-hl, hl)
        ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
        ax3d.set_title("3-D Robot View")

        # ── Draw static frame cube ───────────────────────────────────
        self._draw_frame(ax3d)

        # ── Desired path (faint) ─────────────────────────────────────
        ax3d.plot(
            pos_data[:, 0], pos_data[:, 1], pos_data[:, 2],
            "--", color="lightgray", lw=0.8, alpha=0.6, zorder=1,
        )

        # ── Platform CoM marker ──────────────────────────────────────
        platform_dot, = ax3d.plot([], [], [], "o", ms=8, color="royalblue", zorder=5)

        # ── Cable lines (8 Line3D objects) ───────────────────────────
        cable_lines = []
        for i in range(8):
            ln, = ax3d.plot([], [], [], lw=2.0, solid_capstyle="round")
            cable_lines.append(ln)

        # ── Platform box lines ───────────────────────────────────────
        plat_lines = []
        for _ in range(12):
            ln, = ax3d.plot([], [], [], lw=1.2, color="royalblue")
            plat_lines.append(ln)

        # ── Trail ────────────────────────────────────────────────────
        trail_line, = ax3d.plot([], [], [], "-", lw=1.5, color="limegreen", alpha=0.7)

        # ── Colorbar for cables ──────────────────────────────────────
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.02, label="Tension [N]")

        # ── Tension time-history axes ─────────────────────────────────
        ax_t.set_xlim(t_data[0], t_data[-1])
        ax_t.set_ylim(t_all_min * 0.9, t_all_max * 1.05)
        ax_t.set_xlabel("Time [s]"); ax_t.set_ylabel("Tension [N]")
        ax_t.set_title("Cable Tensions"); ax_t.grid(True, alpha=0.3)
        tension_lines = []
        for i in range(8):
            ln, = ax_t.plot([], [], color=_CABLE_COLORS[i],
                            lw=1.2, label=f"C{i+1}", alpha=0.9)
            tension_lines.append(ln)
        ax_t.legend(ncol=4, fontsize=7, loc="upper right")
        vline_t = ax_t.axvline(t_data[0], color="k", lw=0.8, ls=":")

        # ── Position time-history axes ────────────────────────────────
        pos_min = pos_data.min() * 1.2
        pos_max = pos_data.max() * 1.2
        ax_p.set_xlim(t_data[0], t_data[-1])
        ax_p.set_ylim(pos_min, pos_max)
        ax_p.set_xlabel("Time [s]"); ax_p.set_ylabel("Position [m]")
        ax_p.set_title("Platform Position"); ax_p.grid(True, alpha=0.3)
        pos_lines = []
        for k, (lbl, col) in enumerate(zip(["X","Y","Z"],
                                           ["tab:blue","tab:orange","tab:green"])):
            ln, = ax_p.plot([], [], color=col, lw=1.4, label=lbl)
            pos_lines.append(ln)
        ax_p.legend(fontsize=9)
        vline_p = ax_p.axvline(t_data[0], color="k", lw=0.8, ls=":")

        plt.tight_layout()

        # ── Update function ──────────────────────────────────────────
        def _update(frame):
            p   = pos_data[frame]
            ten = T_data[frame]
            t_now = t_data[frame]

            # Platform marker
            platform_dot.set_data_3d([p[0]], [p[1]], [p[2]])

            # Compute b_world for platform box
            _, _, b_world = self.robot.inverse_kinematics(p)

            # Cable lines (coloured by tension)
            for i in range(8):
                seg_x = [b_world[i, 0], self.robot.B[i, 0]]
                seg_y = [b_world[i, 1], self.robot.B[i, 1]]
                seg_z = [b_world[i, 2], self.robot.B[i, 2]]
                cable_lines[i].set_data_3d(seg_x, seg_y, seg_z)
                cable_lines[i].set_color(cmap(norm(ten[i])))

            # Platform box edges
            plat_segs = _platform_lines(p, b_world)
            for j, seg in enumerate(plat_segs):
                xs = [seg[0][0], seg[1][0]]
                ys = [seg[0][1], seg[1][1]]
                zs = [seg[0][2], seg[1][2]]
                plat_lines[j].set_data_3d(xs, ys, zs)

            # Trail
            lo = max(0, frame - trail_len)
            trail_line.set_data_3d(
                pos_data[lo:frame+1, 0],
                pos_data[lo:frame+1, 1],
                pos_data[lo:frame+1, 2],
            )

            # Tension history
            for i in range(8):
                tension_lines[i].set_data(t_data[:frame+1], T_data[:frame+1, i])
            vline_t.set_xdata([t_now, t_now])

            # Position history
            for k in range(3):
                pos_lines[k].set_data(t_data[:frame+1], pos_data[:frame+1, k])
            vline_p.set_xdata([t_now, t_now])

            return (
                [platform_dot, trail_line]
                + cable_lines
                + plat_lines
                + tension_lines
                + pos_lines
                + [vline_t, vline_p]
            )

        anim = FuncAnimation(
            fig,
            _update,
            frames=N,
            interval=interval_ms,
            blit=False,
        )

        if save_path:
            print(f"  Saving animation to {save_path} …")
            anim.save(save_path, writer="ffmpeg", fps=int(1000 / interval_ms),
                      dpi=90, extra_args=["-vcodec", "libx264"])
            print("  Done.")

        if show:
            plt.show()

        return anim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_frame(self, ax):
        """Draw the static outer cube frame (wireframe)."""
        for (a, b) in _cube_edges(self.robot.frame_size / 2.0):
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="dimgray", lw=1.2, alpha=0.5, zorder=0,
            )
        # Frame anchor spheres
        B = self.robot.B
        ax.scatter(B[:, 0], B[:, 1], B[:, 2],
                   s=35, c="dimgray", zorder=3, depthshade=False)
