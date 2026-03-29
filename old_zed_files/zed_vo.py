"""
zed_vo.py  —  ZED Stereo Visual Odometry for Jetson Autonomous Navigation
==========================================================================
Replaces the monocular pipeline with ZED SDK primitives:

  • TRUE metric scale  — stereo depth, no heuristic guessing
  • Real-time 6-DoF pose — zed.get_position() (ZED positional tracking)
  • IMU fusion          — accelerometer + gyroscope (ZED 2 / ZED 2i / ZED X)
  • Depth sensing       — per-frame depth map for obstacle awareness
  • All navigation data published via a thread-safe NavState dataclass

Outputs (per-frame, printed + saved):
  position (X, Y, Z) [m]       velocity (Vx, Vy, Vz) [m/s]
  orientation (roll, pitch, yaw) [deg]   speed [m/s]
  cumulative distance [m]       distance from origin [m]
  IMU linear acceleration [m/s²]        IMU angular velocity [deg/s]

Requirements
------------
  ZED SDK ≥ 4.x  (https://www.stereolabs.com/developers/release/)
  pyzed           (installed automatically with the ZED SDK)
  opencv-python, numpy, scipy

Usage
-----
  python zed_vo.py                        # live ZED camera
  python zed_vo.py --svo path/to/file.svo # replay an SVO recording
  python zed_vo.py --no-imu               # skip IMU output
  python zed_vo.py --no-plot              # headless / Jetson w/o display
  python zed_vo.py --save-svo out.svo     # record while running
  python zed_vo.py --help                 # all options
"""

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from scipy.signal import savgol_filter

# ── ZED SDK ──────────────────────────────────────────────────────────────────
try:
    import pyzed.sl as sl
except ImportError:
    sys.exit(
        "[ERROR] pyzed not found.\n"
        "Install the ZED SDK from https://www.stereolabs.com/developers/release/\n"
        "then run:  pip install pyzed  (or use the SDK installer)."
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NavState:
    """Thread-safe snapshot of all navigation quantities for one frame."""

    # Pose (from ZED positional tracking — metric, IMU-fused)
    position:    np.ndarray = field(default_factory=lambda: np.zeros(3))   # [m]
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [roll, pitch, yaw] deg
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Velocity / speed (finite-difference on ZED pose)
    velocity:    np.ndarray = field(default_factory=lambda: np.zeros(3))   # [m/s]
    speed:       float = 0.0                                               # [m/s]

    # Distance metrics
    cumulative_distance: float = 0.0   # total path length [m]
    net_displacement:    float = 0.0   # straight-line from origin [m]
    linearity_ratio:     float = 1.0   # net / cumulative

    # IMU (ZED 2 / 2i / X only)
    imu_available:       bool  = False
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [m/s²]
    angular_velocity:    np.ndarray = field(default_factory=lambda: np.zeros(3))  # [deg/s]

    # Frame metadata
    frame_idx:   int   = 0
    timestamp_s: float = 0.0


class DisplacementTracker:
    """
    Accumulates pose updates and exposes navigation statistics.
    Identical logic to the original DisplacementTracker but works with
    metric ZED positions instead of pixel-scale monocular guesses.
    """

    def __init__(self, speed_window: int = 10):
        self.speed_window = speed_window
        self._prev_pos    = None
        self._origin      = None
        self._cum_dist    = 0.0
        self._frame_count = 0
        self._recent_mags = deque(maxlen=speed_window)

        # History (for post-run plots)
        self.positions:        list[np.ndarray] = []
        self.displacement_mags: list[float]     = []
        self.cumulative_dists:  list[float]     = []
        self.net_displacements: list[float]     = []

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
            "step":              step,
            "step_mag":          step_mag,
            "cumulative_distance": self._cum_dist,
            "net_displacement":  net_vec,
            "net_magnitude":     net_mag,
            "smoothed_speed":    smoothed_speed,
            "linearity_ratio":   linearity,
        }

    def summary(self) -> dict:
        mags = np.array(self.displacement_mags)
        return {
            "total_frames":       self._frame_count,
            "total_path_length":  self._cum_dist,
            "final_net_magnitude": self.net_displacements[-1] if self.net_displacements else 0.0,
            "mean_speed":         float(np.mean(mags)) if len(mags) else 0.0,
            "max_speed":          float(np.max(mags))  if len(mags) else 0.0,
            "linearity_ratio":    (self.net_displacements[-1] / self._cum_dist
                                   if self._cum_dist > 1e-9 else 1.0),
        }


class EMASmoothing:
    """Exponential Moving Average — same as original, used for velocity."""
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
    n  = len(traj)
    wl = min(window, n if n % 2 == 1 else n - 1)
    if wl < poly + 2:
        return traj
    return np.stack(
        [savgol_filter(traj[:, i], window_length=wl, polyorder=poly) for i in range(3)],
        axis=1,
    )


# =============================================================================
# ZED Initialisation Helpers
# =============================================================================

def build_init_params(args: argparse.Namespace) -> sl.InitParameters:
    p = sl.InitParameters()

    # Resolution — ZED 2 / 2i / X all support these modes
    res_map = {
        "HD2K":  sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA":   sl.RESOLUTION.VGA,
    }
    p.camera_resolution = res_map.get(args.resolution.upper(), sl.RESOLUTION.HD720)
    p.camera_fps        = args.fps

    # Depth
    p.depth_mode              = sl.DEPTH_MODE.ULTRA  # most accurate
    p.coordinate_units        = sl.UNIT.METER
    p.coordinate_system       = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    p.depth_minimum_distance  = 0.2   # [m] — ZED 2 minimum
    p.depth_maximum_distance  = 20.0  # [m] — tune for your environment

    # SVO input (optional)
    if args.svo:
        p.set_from_svo_file(str(args.svo))
        p.svo_real_time_mode = False

    return p


def build_tracking_params() -> sl.PositionalTrackingParameters:
    tp = sl.PositionalTrackingParameters()
    tp.enable_imu_fusion    = True   # fuse IMU into pose (ZED 2 / 2i / X)
    tp.enable_area_memory   = True   # loop-closure / area learning
    tp.set_as_static        = False
    return tp


# =============================================================================
# Orientation helpers
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
# Live Plotting (optional — disable with --no-plot for headless Jetson)
# =============================================================================

class LivePlotter:
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
               savgol_w: int = 11, savgol_p: int = 3):
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
            ax.set_title(f"ZED VO  |  Frame {frame_idx}  |  Net {net_dist:.2f} m")
            ax.view_init(elev=20, azim=45)
            _set_3d_equal(ax, traj_d)
        else:
            ax.plot(traj_d[:, 0], traj_d[:, 2], "-b", linewidth=2)
            ax.scatter(*traj_d[0, [0, 2]],  c="g", s=60, label="Start")
            ax.scatter(*traj_d[-1, [0, 2]], c="r", s=60, label="Now")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
            ax.set_title(f"ZED VO  |  Frame {frame_idx}  |  Net {net_dist:.2f} m")
            ax.axis("equal")

        ax.legend(); ax.grid(True, alpha=0.3)
        self.plt.pause(0.001)

    def close(self):
        self.plt.ioff()


def _set_3d_equal(ax, traj):
    max_r = np.ptp(traj, axis=0).max() / 2.0 or 1.0
    mid   = traj.mean(axis=0)
    ax.set_xlim(mid[0] - max_r, mid[0] + max_r)
    ax.set_ylim(mid[1] - max_r, mid[1] + max_r)
    ax.set_zlim(mid[2] - max_r, mid[2] + max_r)


# =============================================================================
# Final Analysis Plots
# =============================================================================

def save_analysis_plots(tracker: DisplacementTracker,
                        savgol_w: int = 11, savgol_p: int = 3,
                        view_3d: bool = True):
    import matplotlib.pyplot as plt

    traj   = np.array(tracker.positions)
    if len(traj) < 2:
        print("[WARN] Not enough trajectory points for analysis plots.")
        return

    traj_s = smooth_trajectory(traj, savgol_w, savgol_p)

    mags      = np.array(tracker.displacement_mags)
    cum_dists = np.array(tracker.cumulative_dists)
    net_dists = np.array(tracker.net_displacements)
    time_axis = np.arange(len(mags))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("ZED VO — Navigation Analysis", fontsize=14, fontweight="bold")

    # 1. Position over time
    ax1 = axes[0, 0]
    for i, (lbl, col) in enumerate(zip(["X", "Y", "Z"], ["r", "g", "b"])):
        ax1.plot(time_axis, traj[:, i],   color=col, alpha=0.25, linewidth=1)
        ax1.plot(time_axis, traj_s[:, i], color=col, linewidth=1.5, label=lbl)
    ax1.set_title("Position Over Time"); ax1.set_xlabel("Frame"); ax1.set_ylabel("m")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Per-frame step magnitude (≈ speed between frames)
    ax2 = axes[0, 1]
    ax2.plot(time_axis, mags, color="steelblue", linewidth=1.5)
    ax2.fill_between(time_axis, mags, alpha=0.2, color="steelblue")
    ax2.set_title("Per-Frame Step Size"); ax2.set_xlabel("Frame"); ax2.set_ylabel("m / frame")
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative vs net
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
    ax5.plot(traj[:, 0],   traj[:, 2],   "-b", linewidth=1, alpha=0.25, label="Raw")
    ax5.plot(traj_s[:, 0], traj_s[:, 2], "-b", linewidth=2.5,           label="Smoothed")
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

    # 3-D final trajectory
    fig2 = plt.figure(figsize=(10, 8))
    if view_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax7 = fig2.add_subplot(111, projection="3d")
        ax7.plot(traj[:, 0],   traj[:, 1],   traj[:, 2],   "-b", linewidth=1,   alpha=0.25, label="Raw")
        ax7.plot(traj_s[:, 0], traj_s[:, 1], traj_s[:, 2], "-b", linewidth=2.5,             label="Smoothed")
        ax7.scatter(*traj_s[0],  c="g", s=120, edgecolors="black", label="Start")
        ax7.scatter(*traj_s[-1], c="r", s=120, edgecolors="black", label="End")
        ax7.set_xlabel("X (m)"); ax7.set_ylabel("Y (m)"); ax7.set_zlabel("Z (m)")
        ax7.set_title("Final 3D Trajectory", fontsize=13, fontweight="bold")
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


# =============================================================================
# Main Pipeline
# =============================================================================

def zed_vo(args: argparse.Namespace):
    # ── Open camera ──────────────────────────────────────────────────────────
    zed        = sl.Camera()
    init_params = build_init_params(args)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"[ERROR] Cannot open ZED: {err}\n"
                 "Check USB/CSI connection, SDK version, and permissions.")

    cam_info = zed.get_camera_information()
    model    = str(cam_info.camera_model)
    serial   = cam_info.serial_number
    w        = cam_info.camera_configuration.resolution.width
    h        = cam_info.camera_configuration.resolution.height
    fps_cam  = cam_info.camera_configuration.fps
    print(f"\n{'='*60}")
    print(f"ZED Camera  : {model}  (S/N {serial})")
    print(f"Resolution  : {w}×{h} @ {fps_cam} FPS")
    print(f"Coord system: RIGHT_HANDED_Y_UP  |  Units: metres")
    print(f"{'='*60}\n")

    # ── Optional SVO recording ────────────────────────────────────────────────
    if args.save_svo:
        rec_params = sl.RecordingParameters(
            str(args.save_svo), sl.SVO_COMPRESSION_MODE.H264
        )
        err = zed.enable_recording(rec_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[WARN] SVO recording failed to start: {err}")

    # ── Positional tracking ───────────────────────────────────────────────────
    track_params = build_tracking_params()
    err = zed.enable_positional_tracking(track_params)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        sys.exit(f"[ERROR] Cannot enable positional tracking: {err}")

    # ── ZED runtime params ────────────────────────────────────────────────────
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold  = 50   # depth confidence filter [0–100]
    runtime.texture_confidence_threshold = 100

    # ── Allocate ZED objects ──────────────────────────────────────────────────
    image_left  = sl.Mat()
    depth_map   = sl.Mat()
    pose        = sl.Pose()
    imu_data    = sl.SensorsData()
    #zed_pose    = sl.Transform()

    # ── Navigation state ──────────────────────────────────────────────────────
    tracker     = DisplacementTracker(speed_window=args.speed_window)
    vel_ema     = EMASmoothing(alpha=args.ema_alpha)
    nav         = NavState()
    nav_lock    = Lock()

    # Check IMU availability
    imu_available = (cam_info.camera_model != sl.MODEL.ZED)  # ZED (gen-1) has no IMU
    if not imu_available:
        print("[INFO] ZED gen-1 detected — IMU data not available.")

    # ── Live plotter ──────────────────────────────────────────────────────────
    plotter = None
    if not args.no_plot:
        try:
            plotter = LivePlotter(view_3d=not args.view_2d)
        except Exception as e:
            print(f"[WARN] Could not start live plot: {e}  (try --no-plot)")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"{'Frame':>6}  {'X':>8}  {'Y':>8}  {'Z':>8}  "
          f"{'Spd m/s':>8}  {'Yaw°':>7}  {'Pitch°':>7}  {'Roll°':>7}  "
          f"{'CumDist':>8}  {'NetDist':>8}")
    print("-" * 100)

    frame_idx   = 0
    prev_pos    = None
    prev_time_s = None
    traj_buffer = []   # list of np.ndarray positions for live plot

    try:
        while True:
            # ── Grab frame ───────────────────────────────────────────────────
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                if args.svo:
                    print("\n[INFO] SVO playback complete.")
                    break
                continue

            ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            ts_s  = ts_ns * 1e-9

            dt = (ts_s - prev_time_s) if prev_time_s is not None else (1.0 / max(fps_cam, 1))
            dt = max(dt, 1e-4)

            # ── Pose ─────────────────────────────────────────────────────────
            tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                translation = pose.get_translation()
                pos = np.array([translation.get()[0],
                                translation.get()[1],
                                translation.get()[2]])

                rotation_q = pose.get_orientation().get()   # [ox, oy, oz, ow]
                ox, oy, oz, ow = rotation_q
                # Quaternion → rotation matrix
                R = np.array([
                    [1-2*(oy**2+oz**2),   2*(ox*oy-oz*ow),   2*(ox*oz+oy*ow)],
                    [  2*(ox*oy+oz*ow), 1-2*(ox**2+oz**2),   2*(oy*oz-ox*ow)],
                    [  2*(ox*oz-oy*ow),   2*(oy*oz+ox*ow), 1-2*(ox**2+oy**2)],
                ])
                euler = rotation_matrix_to_euler(R)  # [roll, pitch, yaw] deg

                # Velocity (EMA-smoothed finite difference)
                if prev_pos is not None:
                    raw_vel = (pos - prev_pos) / dt
                    velocity = vel_ema.update(raw_vel)
                else:
                    velocity = np.zeros(3)
                speed_ms = float(np.linalg.norm(velocity))

                # Displacement tracking
                disp = tracker.update(pos, dt)

                # ── IMU ──────────────────────────────────────────────────────
                lin_acc = np.zeros(3)
                ang_vel = np.zeros(3)
                if imu_available and not args.no_imu:
                    zed.get_sensors_data(imu_data, sl.TIME_REFERENCE.IMAGE)
                    imu = imu_data.get_imu_data()
                    la  = imu.get_linear_acceleration()
                    av  = imu.get_angular_velocity()
                    lin_acc = np.array([la[0], la[1], la[2]])
                    ang_vel = np.array([av[0], av[1], av[2]])

                # ── Update NavState (thread-safe) ─────────────────────────────
                with nav_lock:
                    nav.position             = pos
                    nav.orientation          = euler
                    nav.rotation_matrix      = R
                    nav.velocity             = velocity
                    nav.speed                = speed_ms
                    nav.cumulative_distance  = disp["cumulative_distance"]
                    nav.net_displacement     = disp["net_magnitude"]
                    nav.linearity_ratio      = disp["linearity_ratio"]
                    nav.imu_available        = imu_available
                    nav.linear_acceleration  = lin_acc
                    nav.angular_velocity     = ang_vel
                    nav.frame_idx            = frame_idx
                    nav.timestamp_s          = ts_s

                # ── Console output ────────────────────────────────────────────
                if frame_idx % args.verbose_interval == 0:
                    roll, pitch, yaw = euler
                    print(
                        f"{frame_idx:>6}  "
                        f"{pos[0]:>8.3f}  {pos[1]:>8.3f}  {pos[2]:>8.3f}  "
                        f"{speed_ms:>8.3f}  {yaw:>7.1f}  {pitch:>7.1f}  {roll:>7.1f}  "
                        f"{disp['cumulative_distance']:>8.3f}  {disp['net_magnitude']:>8.3f}"
                    )
                    if imu_available and not args.no_imu:
                        print(
                            f"{'':>6}  IMU acc [{lin_acc[0]:+.2f}, {lin_acc[1]:+.2f}, "
                            f"{lin_acc[2]:+.2f}] m/s²   "
                            f"gyro [{ang_vel[0]:+.2f}, {ang_vel[1]:+.2f}, "
                            f"{ang_vel[2]:+.2f}] °/s"
                        )

                traj_buffer.append(pos.copy())
                prev_pos    = pos
                prev_time_s = ts_s

            # ── Optional: show left image with overlay ────────────────────────
            if not args.no_display:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                frame_bgr = image_left.get_data()[:, :, :3].copy()
                _draw_overlay(frame_bgr, nav)
                cv2.imshow("ZED VO", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[INFO] User quit.")
                    break

            # ── Live trajectory plot ──────────────────────────────────────────
            if (plotter is not None
                    and frame_idx % args.plot_interval == 0
                    and len(traj_buffer) > 1):
                traj_arr = np.array(traj_buffer)
                plotter.update(traj_arr, frame_idx,
                               tracker.net_displacements[-1] if tracker.net_displacements else 0.0,
                               savgol_w=args.savgol_window, savgol_p=args.savgol_poly)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    if plotter:
        plotter.close()
    zed.disable_positional_tracking()
    zed.disable_recording()
    zed.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    summary = tracker.summary()
    print(f"\n{'='*60}")
    print("ZED VO — SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Frames processed      : {summary['total_frames']}")
    print(f"Total path length     : {summary['total_path_length']:.4f} m")
    print(f"Net displacement      : {summary['final_net_magnitude']:.4f} m")
    print(f"Mean step speed       : {summary['mean_speed']:.4f} m/frame")
    print(f"Max step speed        : {summary['max_speed']:.4f} m/frame")
    print(f"Linearity ratio       : {summary['linearity_ratio']:.3f}  (1.0 = straight)")
    print(f"{'='*60}\n")

    # Save trajectory
    if tracker.positions:
        traj_arr = np.array(tracker.positions)
        np.savetxt("zed_trajectory.txt", traj_arr,
                   header="X(m) Y(m) Z(m)", comments="# ")
        print("Saved: zed_trajectory.txt")

    # Analysis plots
    if not args.no_plot and len(tracker.positions) > 2:
        save_analysis_plots(tracker,
                            savgol_w=args.savgol_window,
                            savgol_p=args.savgol_poly,
                            view_3d=not args.view_2d)


# =============================================================================
# HUD Overlay on Camera Feed
# =============================================================================

def _draw_overlay(frame: np.ndarray, nav: NavState):
    """Burn key navigation figures onto the camera preview."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    color  = (0, 255, 120)
    small  = 0.52
    med    = 0.60
    lh     = 24

    def put(text, row, scale=small):
        cv2.putText(frame, text, (8, 18 + row * lh), font, scale, color, 1, cv2.LINE_AA)

    p = nav.position
    e = nav.orientation
    put(f"Pos  X:{p[0]:+7.3f}  Y:{p[1]:+7.3f}  Z:{p[2]:+7.3f} m", 0)
    put(f"Rpy  R:{e[0]:+6.1f}  P:{e[1]:+6.1f}  Y:{e[2]:+6.1f} deg", 1)
    put(f"Speed: {nav.speed:.3f} m/s     Frame: {nav.frame_idx}", 2, med)
    put(f"CumDist: {nav.cumulative_distance:.3f} m   NetDist: {nav.net_displacement:.3f} m", 3)
    if nav.imu_available:
        a = nav.linear_acceleration
        put(f"Acc  {a[0]:+.2f}  {a[1]:+.2f}  {a[2]:+.2f} m/s²", 4)
        g = nav.angular_velocity
        put(f"Gyro {g[0]:+.2f}  {g[1]:+.2f}  {g[2]:+.2f} °/s", 5)
    else:
        put("IMU: not available on ZED gen-1", 4)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZED Stereo Visual Odometry — Jetson Autonomous Navigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    parser.add_argument("--svo",       type=Path, default=None,
                        help="Path to .svo file for playback (omit for live camera)")
    parser.add_argument("--save-svo",  type=Path, default=None,
                        help="Record the session to an SVO file")

    # Camera
    parser.add_argument("--resolution", default="HD720",
                        choices=["HD2K", "HD1080", "HD720", "VGA"],
                        help="Camera resolution mode")
    parser.add_argument("--fps",        type=int, default=30,
                        help="Target framerate (must be supported by chosen resolution)")

    # Display / plotting
    parser.add_argument("--no-display",    action="store_true",
                        help="Do not open the camera preview window (headless)")
    parser.add_argument("--no-plot",       action="store_true",
                        help="Disable all matplotlib plotting (headless Jetson)")
    parser.add_argument("--view-2d",       action="store_true",
                        help="Use 2-D (X–Z) trajectory view instead of 3-D")
    parser.add_argument("--plot-interval", type=int, default=10,
                        help="Update live plot every N frames")

    # Navigation / smoothing
    parser.add_argument("--ema-alpha",     type=float, default=0.7,
                        help="EMA alpha for velocity smoothing [0,1]")
    parser.add_argument("--savgol-window", type=int,   default=11,
                        help="Savitzky-Golay window for post-run plots (odd, 0=off)")
    parser.add_argument("--savgol-poly",   type=int,   default=3,
                        help="Savitzky-Golay polynomial order")
    parser.add_argument("--speed-window",  type=int,   default=10,
                        help="Rolling window size for smoothed speed estimate")

    # IMU
    parser.add_argument("--no-imu", action="store_true",
                        help="Skip IMU data retrieval (useful on ZED gen-1)")

    # Verbosity
    parser.add_argument("--verbose-interval", type=int, default=10,
                        help="Print navigation state every N frames")

    args = parser.parse_args()
    zed_vo(args)