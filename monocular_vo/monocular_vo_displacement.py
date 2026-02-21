import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from collections import deque


# ============================
# Camera Intrinsics Utilities
# ============================

def get_camera_matrix(width, height, fov_deg=70):
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg / 2))
    cx = width / 2
    cy = height / 2
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
    ], dtype=np.float64)


# ============================
# Feature Tracking (KLT)
# ============================

def track_features(prev_gray, curr_gray, prev_pts, use_fb_check=True):
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    if use_fb_check:
        prev_reproj, status2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, curr_pts, None, **lk_params)
        fb_error = np.linalg.norm(prev_pts - prev_reproj, axis=2)
        good = (status == 1) & (status2 == 1) & (fb_error < 1.0)
    else:
        good = status == 1

    return prev_pts[good.flatten()].reshape(-1, 2), curr_pts[good.flatten()].reshape(-1, 2)


# ============================
# Pose Estimation
# ============================

def estimate_pose(pts1, pts2, K):
    if len(pts1) < 8:
        return None, None
    E, _ = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None
    _, R, t, _ = cv2.recoverPose(E, pts2, pts1, K)
    return R, t


# ============================
# Scale Heuristic
# ============================

def estimate_scale(prev_pts, curr_pts):
    dists = np.linalg.norm(curr_pts - prev_pts, axis=1)
    return max(np.median(dists), 1e-3)


# ============================
# EMA Smoother
# ============================

class EMASmoothing:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.smoothed = None

    def update(self, value):
        if self.smoothed is None:
            self.smoothed = value.copy()
        else:
            self.smoothed = self.alpha * value + (1 - self.alpha) * self.smoothed
        return self.smoothed.copy()


# ============================
# Real-Time Displacement Tracker
# ============================

class DisplacementTracker:
    """
    Tracks camera displacement in real time as poses are accumulated.

    Maintains:
      - per-frame displacement (the step from the last pose to the current one)
      - cumulative path length (sum of all step magnitudes)
      - net displacement vector from the origin
      - a rolling window of recent displacements for smoothed speed estimation
    """

    def __init__(self, smoothing_window=10):
        """
        Args:
            smoothing_window: Number of recent frames used to compute a
                              rolling-average speed estimate.
        """
        self.smoothing_window = smoothing_window

        # Running state
        self.prev_t = np.zeros(3)              # position at the previous frame
        self.origin  = np.zeros(3)             # position at frame 0

        # Accumulators
        self.cumulative_distance = 0.0         # total path length so far
        self.frame_count = 0

        # Per-frame records (grown each frame, O(N) memory)
        self.displacements      = []           # per-frame 3D displacement vectors
        self.displacement_mags  = []           # per-frame scalar magnitudes
        self.cumulative_dists   = []           # cumulative distance at each frame
        self.net_displacements  = []           # distance from origin at each frame

        # Rolling window for speed estimate
        self._recent_mags = deque(maxlen=smoothing_window)

    # ------------------------------------------------------------------
    def update(self, current_t: np.ndarray):
        """
        Call once per frame with the current world-space position vector.

        Args:
            current_t: (3,) array — the camera's current position.

        Returns:
            dict with keys:
                frame_displacement      (3,)  — XYZ step this frame
                frame_magnitude         float — Euclidean length of that step
                cumulative_distance     float — total path length so far
                net_displacement        (3,)  — vector from origin to now
                net_magnitude           float — straight-line distance from origin
                smoothed_speed          float — rolling-average step magnitude
                linearity_ratio         float — net / cumulative (1 = straight line)
        """
        current_t = np.asarray(current_t, dtype=np.float64).flatten()

        if self.frame_count == 0:
            self.origin = current_t.copy()

        # --- Per-frame displacement ---
        frame_disp = current_t - self.prev_t
        frame_mag  = float(np.linalg.norm(frame_disp))

        # --- Cumulative distance (path length) ---
        self.cumulative_distance += frame_mag

        # --- Net displacement from origin ---
        net_disp = current_t - self.origin
        net_mag  = float(np.linalg.norm(net_disp))

        # --- Rolling speed ---
        self._recent_mags.append(frame_mag)
        smoothed_speed = float(np.mean(self._recent_mags))

        # --- Linearity ratio ---
        linearity = (net_mag / self.cumulative_distance
                     if self.cumulative_distance > 1e-9 else 1.0)

        # Store history
        self.displacements.append(frame_disp.copy())
        self.displacement_mags.append(frame_mag)
        self.cumulative_dists.append(self.cumulative_distance)
        self.net_displacements.append(net_mag)

        # Advance state
        self.prev_t = current_t.copy()
        self.frame_count += 1

        return {
            "frame_displacement":  frame_disp,
            "frame_magnitude":     frame_mag,
            "cumulative_distance": self.cumulative_distance,
            "net_displacement":    net_disp,
            "net_magnitude":       net_mag,
            "smoothed_speed":      smoothed_speed,
            "linearity_ratio":     linearity,
        }

    # ------------------------------------------------------------------
    def summary(self):
        """Return a dict of overall statistics after processing is complete."""
        mags = np.array(self.displacement_mags)
        return {
            "total_frames":         self.frame_count,
            "total_path_length":    self.cumulative_distance,
            "final_net_magnitude":  self.net_displacements[-1] if self.net_displacements else 0.0,
            "mean_speed":           float(np.mean(mags)) if len(mags) else 0.0,
            "max_speed":            float(np.max(mags))  if len(mags) else 0.0,
            "linearity_ratio":      self.net_displacements[-1] / self.cumulative_distance
                                    if self.cumulative_distance > 1e-9 else 1.0,
        }


# ============================
# Trajectory Smoothing (post)
# ============================

def smooth_trajectory(trajectory, window_length=11, polyorder=3):
    n = len(trajectory)
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < polyorder + 2:
        return trajectory
    return np.stack([
        savgol_filter(trajectory[:, i], window_length=wl, polyorder=polyorder)
        for i in range(3)
    ], axis=1)


# ============================
# Main Pipeline
# ============================

def monocular_vo_displacement(
        video_path=None,
        scale_factor=1.0,
        max_features=500,
        min_features=200,
        use_fb_check=True,
        frame_skip=1,
        plot_interval=10,
        redetect_interval=5,
        view_3d=True,
        ema_alpha=0.7,
        savgol_window=15,
        savgol_poly=3,
        speed_window=10,
        verbose_interval=10):
    """
    Args:
        speed_window:     Rolling window size for real-time speed estimate.
        verbose_interval: Print displacement stats every N frames.
    """

    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    if not cap.isOpened():
        print("Failed to open video source.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    ret, frame = cap.read()
    if not ret:
        print("Cannot read first frame.")
        return

    if scale_factor != 1.0:
        frame  = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        width  = int(width  * scale_factor)
        height = int(height * scale_factor)

    K = get_camera_matrix(width, height)
    print("Camera matrix K:\n", K)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts  = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_features,
        qualityLevel=0.01, minDistance=7, blockSize=7
    )

    # Pose state
    R = np.eye(3)
    t = np.zeros((3, 1))

    # Smoothers / trackers
    ema     = EMASmoothing(alpha=ema_alpha)
    tracker = DisplacementTracker(smoothing_window=speed_window)

    # Trajectory storage
    trajectory = (np.zeros((total_frames, 3)) if total_frames > 0
                  else np.zeros((10_000, 3)))
    traj_idx = 0

    # Live plot
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d') if view_3d else fig.add_subplot(111)

    frame_idx       = 0
    redetect_counter = 0

    print("\n--- Real-Time Displacement Output ---")
    print(f"{'Frame':>6}  {'Step':>8}  {'CumDist':>9}  {'NetDist':>9}  "
          f"{'Speed(avg)':>10}  {'Linearity':>9}  {'NetVec (X,Y,Z)'}")
    print("-" * 90)

    while True:
        for _ in range(frame_skip):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break

        if scale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prev_pts_tracked, curr_pts_tracked = track_features(
            prev_gray, curr_gray, prev_pts, use_fb_check
        )

        if len(prev_pts_tracked) > 8:
            R_rel, t_rel = estimate_pose(prev_pts_tracked, curr_pts_tracked, K)
            if R_rel is not None:
                scale = estimate_scale(prev_pts_tracked, curr_pts_tracked)

                raw_increment      = scale * (R @ t_rel)
                smoothed_increment = ema.update(raw_increment)

                t = t + smoothed_increment
                R = R_rel @ R

                trajectory[traj_idx] = t.flatten()
                traj_idx += 1

                # ── Real-time displacement update ──────────────────────────
                disp = tracker.update(t.flatten())

                if frame_idx % verbose_interval == 0:
                    nv = disp["net_displacement"]
                    print(
                        f"{frame_idx:>6}  "
                        f"{disp['frame_magnitude']:>8.4f}  "
                        f"{disp['cumulative_distance']:>9.4f}  "
                        f"{disp['net_magnitude']:>9.4f}  "
                        f"{disp['smoothed_speed']:>10.4f}  "
                        f"{disp['linearity_ratio']:>9.3f}  "
                        f"({nv[0]:+.3f}, {nv[1]:+.3f}, {nv[2]:+.3f})"
                    )

        # Feature re-detection
        redetect_counter += 1
        if (redetect_counter % redetect_interval == 0
                and len(curr_pts_tracked) < min_features):
            curr_pts_tracked = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=max_features,
                qualityLevel=0.01, minDistance=7, blockSize=7
            )
            print(f"  [Re-detected features at frame {frame_idx}]")

        prev_gray = curr_gray
        prev_pts  = curr_pts_tracked.reshape(-1, 1, 2)
        frame_idx += 1

        # Live trajectory plot
        if frame_idx % plot_interval == 0 and traj_idx > 1:
            traj = trajectory[:traj_idx]
            traj_display = (smooth_trajectory(traj, savgol_window, savgol_poly)
                            if savgol_window > 1 and traj_idx >= savgol_window
                            else traj)
            ax.clear()

            if view_3d:
                ax.plot(traj_display[:, 0], traj_display[:, 1], traj_display[:, 2],
                        '-b', linewidth=2)
                ax.scatter(*traj_display[0],  c='g', s=50, marker='o')
                ax.scatter(*traj_display[-1], c='r', s=50, marker='o')
                ax.set_title(f"VO Trajectory  |  Frame {frame_idx}  "
                             f"|  Net: {tracker.net_displacements[-1]:.3f}")
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

                max_r = np.array([
                    np.ptp(traj_display[:, 0]),
                    np.ptp(traj_display[:, 1]),
                    np.ptp(traj_display[:, 2])
                ]).max() / 2.0 or 1.0
                mid = traj_display.mean(axis=0)
                ax.set_xlim(mid[0]-max_r, mid[0]+max_r)
                ax.set_ylim(mid[1]-max_r, mid[1]+max_r)
                ax.set_zlim(mid[2]-max_r, mid[2]+max_r)
                ax.view_init(elev=20, azim=45)
            else:
                ax.plot(traj_display[:, 0], traj_display[:, 2], '-b', linewidth=2)
                ax.scatter(*traj_display[0, [0, 2]],  c='g', s=50)
                ax.scatter(*traj_display[-1, [0, 2]], c='r', s=50)
                ax.set_title(f"VO Trajectory  |  Frame {frame_idx}  "
                             f"|  Net: {tracker.net_displacements[-1]:.3f}")
                ax.set_xlabel("X"); ax.set_ylabel("Z")
                ax.axis('equal')

            ax.grid(True, alpha=0.3)
            plt.pause(0.001)

    cap.release()
    plt.ioff()

    # ── Final summary ──────────────────────────────────────────────────
    trajectory = trajectory[:traj_idx]
    summary    = tracker.summary()

    print("\n" + "=" * 50)
    print("DISPLACEMENT SUMMARY")
    print("=" * 50)
    print(f"Total frames processed : {summary['total_frames']}")
    print(f"Total path length      : {summary['total_path_length']:.4f} units")
    print(f"Net displacement       : {summary['final_net_magnitude']:.4f} units")
    print(f"Mean speed             : {summary['mean_speed']:.4f} units/frame")
    print(f"Max speed              : {summary['max_speed']:.4f} units/frame")
    print(f"Linearity ratio        : {summary['linearity_ratio']:.3f}  "
          f"(1.0 = perfectly straight)")
    print("=" * 50)

    np.savetxt("trajectory_displacement.txt", trajectory)
    print("Saved trajectory to trajectory_displacement.txt")

    # ── Displacement plots ─────────────────────────────────────────────
    if len(trajectory) > 1:
        trajectory_smooth = (smooth_trajectory(trajectory, savgol_window, savgol_poly)
                             if savgol_window > 1 and len(trajectory) >= savgol_window
                             else trajectory)

        time_axis  = np.arange(len(tracker.displacement_mags))
        mags       = np.array(tracker.displacement_mags)
        cum_dists  = np.array(tracker.cumulative_dists)
        net_dists  = np.array(tracker.net_displacements)
        disps      = np.array(tracker.displacements)

        fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig2.suptitle("Real-Time Displacement Analysis", fontsize=14, fontweight='bold')

        # 1. Per-frame displacement magnitude
        ax1 = axes[0, 0]
        ax1.plot(time_axis, mags, color='steelblue', linewidth=1.5)
        ax1.fill_between(time_axis, mags, alpha=0.2, color='steelblue')
        ax1.set_title("Per-Frame Displacement (step size)")
        ax1.set_xlabel("Frame"); ax1.set_ylabel("Magnitude (units)")
        ax1.grid(True, alpha=0.3)

        # 2. Per-frame displacement components
        ax2 = axes[0, 1]
        ax2.plot(time_axis, disps[:, 0], 'r-', linewidth=1.2, label='ΔX')
        ax2.plot(time_axis, disps[:, 1], 'g-', linewidth=1.2, label='ΔY')
        ax2.plot(time_axis, disps[:, 2], 'b-', linewidth=1.2, label='ΔZ')
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax2.set_title("Per-Frame Displacement Components")
        ax2.set_xlabel("Frame"); ax2.set_ylabel("Displacement (units)")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        # 3. Cumulative distance vs net displacement
        ax3 = axes[0, 2]
        ax3.plot(time_axis, cum_dists, color='orange', linewidth=2, label='Path length')
        ax3.plot(time_axis, net_dists, color='teal',   linewidth=2, label='Net distance')
        ax3.fill_between(time_axis, net_dists, cum_dists, alpha=0.15, color='gray',
                         label='Wasted motion')
        ax3.set_title("Cumulative Distance vs Net Displacement")
        ax3.set_xlabel("Frame"); ax3.set_ylabel("Distance (units)")
        ax3.legend(); ax3.grid(True, alpha=0.3)

        # 4. Linearity ratio over time
        linearity = np.where(cum_dists > 1e-9, net_dists / cum_dists, 1.0)
        ax4 = axes[1, 0]
        ax4.plot(time_axis, linearity, color='purple', linewidth=1.5)
        ax4.axhline(1.0, color='black', linewidth=0.5, linestyle='--', label='Perfect linear')
        ax4.set_ylim(0, 1.05)
        ax4.set_title("Linearity Ratio (net / path)")
        ax4.set_xlabel("Frame"); ax4.set_ylabel("Ratio")
        ax4.legend(); ax4.grid(True, alpha=0.3)

        # 5. Rolling average speed
        roll_speed = np.convolve(mags, np.ones(speed_window) / speed_window, mode='same')
        ax5 = axes[1, 1]
        ax5.plot(time_axis, mags,       color='steelblue', linewidth=1, alpha=0.4, label='Raw')
        ax5.plot(time_axis, roll_speed, color='steelblue', linewidth=2,            label=f'Rolling avg ({speed_window}f)')
        ax5.set_title("Speed (rolling average)")
        ax5.set_xlabel("Frame"); ax5.set_ylabel("Speed (units/frame)")
        ax5.legend(); ax5.grid(True, alpha=0.3)

        # 6. Final trajectory (raw + smoothed)
        ax6 = axes[1, 2]
        ax6.plot(trajectory[:, 0],        trajectory[:, 2],        '-b', linewidth=1,   alpha=0.25, label='Raw')
        ax6.plot(trajectory_smooth[:, 0], trajectory_smooth[:, 2], '-b', linewidth=2.5,             label='Smoothed')
        ax6.scatter(trajectory_smooth[0, 0],  trajectory_smooth[0, 2],  c='g', s=80, zorder=5, label='Start')
        ax6.scatter(trajectory_smooth[-1, 0], trajectory_smooth[-1, 2], c='r', s=80, zorder=5, label='End')
        ax6.set_title("Top-Down Trajectory (X-Z)")
        ax6.set_xlabel("X"); ax6.set_ylabel("Z")
        ax6.legend(); ax6.axis('equal'); ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("displacement_analysis.png", dpi=150, bbox_inches='tight')
        print("Saved displacement analysis to displacement_analysis.png")

        # Final 3D trajectory plot
        fig3 = plt.figure(figsize=(10, 8))
        if view_3d:
            ax7 = fig3.add_subplot(111, projection='3d')
            ax7.plot(trajectory[:, 0],        trajectory[:, 1],        trajectory[:, 2],
                     '-b', linewidth=1, alpha=0.25, label='Raw')
            ax7.plot(trajectory_smooth[:, 0], trajectory_smooth[:, 1], trajectory_smooth[:, 2],
                     '-b', linewidth=2.5, label='Smoothed')
            ax7.scatter(*trajectory_smooth[0],  c='g', s=120, edgecolors='black', label='Start')
            ax7.scatter(*trajectory_smooth[-1], c='r', s=120, edgecolors='black', label='End')
            ax7.set_title("Final 3D Trajectory", fontsize=13, fontweight='bold')
            ax7.set_xlabel("X"); ax7.set_ylabel("Y"); ax7.set_zlabel("Z")
            ax7.legend(); ax7.grid(True, alpha=0.3)
            ax7.view_init(elev=20, azim=45)
        else:
            ax7 = fig3.add_subplot(111)
            ax7.plot(trajectory[:, 0],        trajectory[:, 2],        '-b', linewidth=1, alpha=0.25, label='Raw')
            ax7.plot(trajectory_smooth[:, 0], trajectory_smooth[:, 2], '-b', linewidth=2.5,           label='Smoothed')
            ax7.scatter(trajectory_smooth[0, 0],  trajectory_smooth[0, 2],  c='g', s=100, label='Start')
            ax7.scatter(trajectory_smooth[-1, 0], trajectory_smooth[-1, 2], c='r', s=100, label='End')
            ax7.set_title("Final Trajectory (X-Z)"); ax7.set_xlabel("X"); ax7.set_ylabel("Z")
            ax7.legend(); ax7.axis('equal'); ax7.grid(True, alpha=0.3)

        plt.savefig("trajectory_plot.png", dpi=150, bbox_inches='tight')
        print("Saved trajectory plot to trajectory_plot.png")
        plt.show()


# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monocular VO with real-time displacement computation"
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to video file (omit for webcam)")
    parser.add_argument("--scale",        type=float, default=1.0)
    parser.add_argument("--features",     type=int,   default=500)
    parser.add_argument("--min-features", type=int,   default=200)
    parser.add_argument("--no-fb",        action="store_true")
    parser.add_argument("--skip",         type=int,   default=1)
    parser.add_argument("--plot-interval",type=int,   default=10)
    parser.add_argument("--2d",           action="store_true")
    parser.add_argument("--ema-alpha",    type=float, default=0.7,
                        help="EMA smoothing factor [0,1]")
    parser.add_argument("--savgol-window",type=int,   default=15,
                        help="Savitzky-Golay window (odd, 0=off)")
    parser.add_argument("--savgol-poly",  type=int,   default=3)
    parser.add_argument("--speed-window", type=int,   default=10,
                        help="Rolling window for real-time speed estimate")
    parser.add_argument("--verbose",      type=int,   default=10,
                        help="Print displacement every N frames")

    args = parser.parse_args()

    monocular_vo_displacement(
        video_path      = args.video,
        scale_factor    = args.scale,
        max_features    = args.features,
        min_features    = args.min_features,
        use_fb_check    = not args.no_fb,
        frame_skip      = args.skip,
        plot_interval   = args.plot_interval,
        view_3d         = not args.__dict__["2d"],
        ema_alpha       = args.ema_alpha,
        savgol_window   = args.savgol_window,
        savgol_poly     = args.savgol_poly,
        speed_window    = args.speed_window,
        verbose_interval= args.verbose,
    )