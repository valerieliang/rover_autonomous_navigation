"""
zed_vo_core.py  —  Visual Odometry helpers (imported by zed_slam_main.py)
=========================================================================
Contains all VO-specific logic that does NOT touch the ZED camera handle:
  • NavState          — thread-safe navigation snapshot dataclass
  • DisplacementTracker — accumulates pose updates, exposes path statistics
  • EMASmoothing       — exponential moving average for velocity
  • smooth_trajectory  — Savitzky-Golay smoothing for post-run plots
  • rotation_matrix_to_euler — SO(3) → roll/pitch/yaw
  • LivePlotter        — matplotlib real-time trajectory window
  • save_analysis_plots — post-session analysis figure suite

Nothing here opens a camera or calls sl.Camera.  The main loop in
zed_slam_main.py owns the ZED handle and passes poses + timestamps in.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class NavState:
    """Thread-safe snapshot of all navigation quantities for one frame."""

    # Pose (metric, IMU-fused via ZED positional tracking)
    position:         np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation:      np.ndarray = field(default_factory=lambda: np.zeros(3))   # [roll, pitch, yaw] deg
    rotation_matrix:  np.ndarray = field(default_factory=lambda: np.eye(3))

    # Velocity / speed (finite-difference on ZED pose)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [m/s]
    speed:    float = 0.0                                                # scalar [m/s]

    # Distance metrics
    cumulative_distance: float = 0.0   # total path length [m]
    net_displacement:    float = 0.0   # straight-line from origin [m]
    linearity_ratio:     float = 1.0   # net / cumulative

    # IMU (ZED 2 / 2i / X only)
    imu_available:        bool        = False
    linear_acceleration:  np.ndarray  = field(default_factory=lambda: np.zeros(3))  # [m/s²]
    angular_velocity:     np.ndarray  = field(default_factory=lambda: np.zeros(3))  # [deg/s]

    # Frame metadata
    frame_idx:   int   = 0
    timestamp_s: float = 0.0


# =============================================================================
# Displacement / path tracking
# =============================================================================

class DisplacementTracker:
    """
    Accumulates pose updates from ZED positional tracking and exposes
    navigation statistics (cumulative path length, net displacement, speed).
    """

    def __init__(self, speed_window: int = 10):
        self.speed_window = speed_window
        self._prev_pos    = None
        self._origin      = None
        self._cum_dist    = 0.0
        self._frame_count = 0
        self._recent_mags: deque[float] = deque(maxlen=speed_window)

        # Full history for post-run plots
        self.positions:          list[np.ndarray] = []
        self.displacement_mags:  list[float]      = []
        self.cumulative_dists:   list[float]      = []
        self.net_displacements:  list[float]      = []

    def update(self, pos: np.ndarray, dt: float) -> dict:
        pos = np.asarray(pos, dtype=np.float64).flatten()

        if self._frame_count == 0:
            self._origin   = pos.copy()
            self._prev_pos = pos.copy()

        step     = pos - self._prev_pos
        step_mag = float(np.linalg.norm(step))
        self._cum_dist += step_mag

        net_vec = pos - self._origin
        net_mag = float(np.linalg.norm(net_vec))

        self._recent_mags.append(step_mag)
        smoothed_speed = float(np.mean(self._recent_mags)) / max(dt, 1e-6)

        linearity = net_mag / self._cum_dist if self._cum_dist > 1e-9 else 1.0

        self.positions.append(pos.copy())
        self.displacement_mags.append(step_mag)
        self.cumulative_dists.append(self._cum_dist)
        self.net_displacements.append(net_mag)

        self._prev_pos = pos.copy()
        self._frame_count += 1

        return {
            "step":               step,
            "step_mag":           step_mag,
            "cumulative_distance": self._cum_dist,
            "net_displacement":   net_vec,
            "net_magnitude":      net_mag,
            "smoothed_speed":     smoothed_speed,
            "linearity_ratio":    linearity,
        }

    def summary(self) -> dict:
        mags = np.array(self.displacement_mags)
        return {
            "total_frames":         self._frame_count,
            "total_path_length":    self._cum_dist,
            "final_net_magnitude":  self.net_displacements[-1] if self.net_displacements else 0.0,
            "mean_speed":           float(np.mean(mags)) if len(mags) else 0.0,
            "max_speed":            float(np.max(mags))  if len(mags) else 0.0,
            "linearity_ratio":      (self.net_displacements[-1] / self._cum_dist
                                     if self._cum_dist > 1e-9 else 1.0),
        }


# =============================================================================
# Smoothing utilities
# =============================================================================

class EMASmoothing:
    """Exponential Moving Average — used for velocity smoothing."""

    def __init__(self, alpha: float = 0.7):
        self.alpha    = alpha
        self.smoothed = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self.smoothed is None:
            self.smoothed = value.copy()
        else:
            self.smoothed = self.alpha * value + (1 - self.alpha) * self.smoothed
        return self.smoothed.copy()


def smooth_trajectory(traj: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to an (N, 3) trajectory array."""
    n  = len(traj)
    wl = min(window, n if n % 2 == 1 else n - 1)
    if wl < poly + 2:
        return traj
    return np.stack(
        [savgol_filter(traj[:, i], window_length=wl, polyorder=poly)
         for i in range(3)],
        axis=1,
    )


# =============================================================================
# Orientation helper
# =============================================================================

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Return [roll, pitch, yaw] in degrees from a 3×3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2( R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2( R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.degrees([roll, pitch, yaw])


# =============================================================================
# Live plotting (optional — disable with --no-plot for headless Jetson)
# =============================================================================

class LivePlotter:
    """Real-time matplotlib trajectory window, updated every N frames."""

    def __init__(self, view_3d: bool = True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        self.plt    = plt
        self.view3d = view_3d

        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax  = (self.fig.add_subplot(111, projection="3d")
                    if view_3d else self.fig.add_subplot(111))

    def update(self, traj: np.ndarray, frame_idx: int, net_dist: float,
               savgol_w: int = 11, savgol_p: int = 3) -> None:
        if len(traj) < 2:
            return
        traj_d = (smooth_trajectory(traj, savgol_w, savgol_p)
                  if savgol_w > 1 and len(traj) >= savgol_w else traj)
        ax = self.ax
        ax.clear()

        if self.view3d:
            ax.plot(traj_d[:, 0], traj_d[:, 1], traj_d[:, 2], "-b", linewidth=2)
            ax.scatter(*traj_d[0],  c="g", s=60, marker="o", label="Start")
            ax.scatter(*traj_d[-1], c="r", s=60, marker="o", label="Now")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
            ax.set_title(f"ZED SLAM  |  Frame {frame_idx}  |  Net {net_dist:.2f} m")
            ax.view_init(elev=20, azim=45)
            _set_3d_equal(ax, traj_d)
        else:
            ax.plot(traj_d[:, 0], traj_d[:, 2], "-b", linewidth=2)
            ax.scatter(*traj_d[0, [0, 2]],  c="g", s=60, label="Start")
            ax.scatter(*traj_d[-1, [0, 2]], c="r", s=60, label="Now")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
            ax.set_title(f"ZED SLAM  |  Frame {frame_idx}  |  Net {net_dist:.2f} m")
            ax.axis("equal")

        ax.legend(); ax.grid(True, alpha=0.3)
        self.plt.pause(0.001)

    def close(self) -> None:
        self.plt.ioff()


def _set_3d_equal(ax, traj: np.ndarray) -> None:
    max_r = np.ptp(traj, axis=0).max() / 2.0 or 1.0
    mid   = traj.mean(axis=0)
    ax.set_xlim(mid[0] - max_r, mid[0] + max_r)
    ax.set_ylim(mid[1] - max_r, mid[1] + max_r)
    ax.set_zlim(mid[2] - max_r, mid[2] + max_r)


# =============================================================================
# Post-session analysis plots
# =============================================================================

def save_analysis_plots(tracker: DisplacementTracker,
                        savgol_w: int = 11, savgol_p: int = 3,
                        view_3d: bool = True) -> None:
    import matplotlib.pyplot as plt

    traj = np.array(tracker.positions)
    if len(traj) < 2:
        print("[WARN] Not enough trajectory points for analysis plots.")
        return

    traj_s    = smooth_trajectory(traj, savgol_w, savgol_p)
    mags      = np.array(tracker.displacement_mags)
    cum_dists = np.array(tracker.cumulative_dists)
    net_dists = np.array(tracker.net_displacements)
    time_axis = np.arange(len(mags))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("ZED SLAM — Navigation Analysis", fontsize=14, fontweight="bold")

    # 1. Position over time
    ax1 = axes[0, 0]
    for i, (lbl, col) in enumerate(zip(["X", "Y", "Z"], ["r", "g", "b"])):
        ax1.plot(time_axis, traj[:, i],   color=col, alpha=0.25, linewidth=1)
        ax1.plot(time_axis, traj_s[:, i], color=col, linewidth=1.5, label=lbl)
    ax1.set_title("Position Over Time"); ax1.set_xlabel("Frame"); ax1.set_ylabel("m")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Per-frame step magnitude (≈ inter-frame speed)
    ax2 = axes[0, 1]
    ax2.plot(time_axis, mags, color="steelblue", linewidth=1.5)
    ax2.fill_between(time_axis, mags, alpha=0.2, color="steelblue")
    ax2.set_title("Per-Frame Step Size"); ax2.set_xlabel("Frame"); ax2.set_ylabel("m / frame")
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative vs net distance
    ax3 = axes[0, 2]
    ax3.plot(time_axis, cum_dists, color="orange", linewidth=2, label="Path length")
    ax3.plot(time_axis, net_dists, color="teal",   linewidth=2, label="Net dist")
    ax3.fill_between(time_axis, net_dists, cum_dists, alpha=0.15, color="gray",
                     label="Wasted motion")
    ax3.set_title("Cumulative vs Net Distance"); ax3.set_xlabel("Frame"); ax3.set_ylabel("m")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # 4. Linearity ratio
    lin = np.where(cum_dists > 1e-9, net_dists / cum_dists, 1.0)
    ax4 = axes[1, 0]
    ax4.plot(time_axis, lin, color="purple", linewidth=1.5)
    ax4.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("Linearity Ratio"); ax4.set_xlabel("Frame"); ax4.set_ylabel("ratio")
    ax4.grid(True, alpha=0.3)

    # 5. XZ top-down trajectory
    ax5 = axes[1, 1]
    ax5.plot(traj[:, 0],   traj[:, 2],   "-b", linewidth=1,   alpha=0.25, label="Raw")
    ax5.plot(traj_s[:, 0], traj_s[:, 2], "-b", linewidth=2.5,             label="Smoothed")
    ax5.scatter(traj_s[0, 0],  traj_s[0, 2],  c="g", s=80, zorder=5)
    ax5.scatter(traj_s[-1, 0], traj_s[-1, 2], c="r", s=80, zorder=5)
    ax5.set_title("Top-Down (X–Z)"); ax5.set_xlabel("X (m)"); ax5.set_ylabel("Z (m)")
    ax5.legend(); ax5.axis("equal"); ax5.grid(True, alpha=0.3)

    # 6. Height (Y) over time
    ax6 = axes[1, 2]
    ax6.plot(time_axis, traj[:, 1],   color="green", alpha=0.2, linewidth=1)
    ax6.plot(time_axis, traj_s[:, 1], color="green", linewidth=2)
    ax6.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax6.fill_between(time_axis, traj_s[:, 1], alpha=0.2, color="green")
    ax6.set_title("Height (Y) Over Time"); ax6.set_xlabel("Frame"); ax6.set_ylabel("m")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("zed_displacement_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved: zed_displacement_analysis.png")

    # Final 3-D trajectory
    fig2 = plt.figure(figsize=(10, 8))
    if view_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax7 = fig2.add_subplot(111, projection="3d")
        ax7.plot(traj[:, 0],   traj[:, 1],   traj[:, 2],   "-b", linewidth=1,   alpha=0.25, label="Raw")
        ax7.plot(traj_s[:, 0], traj_s[:, 1], traj_s[:, 2], "-b", linewidth=2.5,             label="Smoothed")
        ax7.scatter(*traj_s[0],  c="g", s=120, edgecolors="black", label="Start")
        ax7.scatter(*traj_s[-1], c="r", s=120, edgecolors="black", label="End")
        ax7.set_xlabel("X (m)"); ax7.set_ylabel("Y (m)"); ax7.set_zlabel("Z (m)")
        ax7.set_title("Final 3-D Trajectory", fontsize=13, fontweight="bold")
        ax7.legend(); ax7.grid(True, alpha=0.3); ax7.view_init(elev=20, azim=45)
        _set_3d_equal(ax7, traj_s)
    else:
        ax7 = fig2.add_subplot(111)
        ax7.plot(traj[:, 0],   traj[:, 2],   "-b", linewidth=1, alpha=0.25, label="Raw")
        ax7.plot(traj_s[:, 0], traj_s[:, 2], "-b", linewidth=2.5,           label="Smoothed")
        ax7.scatter(traj_s[0, 0],  traj_s[0, 2],  c="g", s=100)
        ax7.scatter(traj_s[-1, 0], traj_s[-1, 2], c="r", s=100)
        ax7.set_xlabel("X (m)"); ax7.set_ylabel("Z (m)")
        ax7.set_title("Final Trajectory (X–Z)"); ax7.axis("equal"); ax7.grid(True, alpha=0.3)

    plt.savefig("zed_trajectory_plot.png", dpi=150, bbox_inches="tight")
    print("Saved: zed_trajectory_plot.png")
    plt.show()
