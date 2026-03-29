"""
zed_slam_main.py  —  ZED Camera SLAM: Unified Entry Point
==========================================================
Runs Visual Odometry (VO) and Scene Understanding (wall/hallway detection +
line-of-sight object validation) together in a single loop, sharing one camera
handle and one grab() call per frame.

Modules
-------
  zed_vo_core.py              — NavState, DisplacementTracker, EMASmoothing,
                                orientation helpers, live plotter, analysis plots
  zed_scene_core.py           — Config, SceneState dataclasses, all analysis
                                functions (detect_walls, detect_hallway,
                                probe_line_of_sight, find_forward_clusters),
                                draw_overlay, colorise_depth

Usage
-----
  python zed_slam_main.py                         # live camera, all features on
  python zed_slam_main.py --svo path/to/file.svo  # replay SVO recording
  python zed_slam_main.py --no-display            # headless (no OpenCV windows)
  python zed_slam_main.py --no-plot               # skip matplotlib (Jetson)
  python zed_slam_main.py --no-imu               # skip IMU (ZED gen-1)
  python zed_slam_main.py --save-svo out.svo      # record while running
  python zed_slam_main.py --view-2d               # 2-D trajectory view
  python zed_slam_main.py --help                  # all options
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

# ── ZED SDK ───────────────────────────────────────────────────────────────────
try:
    import pyzed.sl as sl
except ImportError:
    sys.exit(
        "[ERROR] pyzed not found.\n"
        "Install the ZED SDK from https://www.stereolabs.com/developers/release/"
    )

# ── Local modules (must be in the same directory) ────────────────────────────
try:
    from zed_vo_core import (
        NavState,
        DisplacementTracker,
        EMASmoothing,
        rotation_matrix_to_euler,
        LivePlotter,
        save_analysis_plots,
        smooth_trajectory,
    )
    from zed_scene_core import (
        SceneConfig,
        SceneState,
        preprocess_depth,
        detect_walls,
        detect_hallway,
        probe_line_of_sight,
        find_forward_clusters,
        draw_scene_overlay,
        colorise_depth,
    )
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import helper module: {e}\n"
        "Make sure zed_vo_core.py and zed_scene_core.py are in the same directory."
    )


# =============================================================================
# Camera / tracking initialisation helpers
# =============================================================================

def _build_init_params(args: argparse.Namespace) -> sl.InitParameters:
    p = sl.InitParameters()

    res_map = {
        "HD2K":  sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA":   sl.RESOLUTION.VGA,
    }
    p.camera_resolution = res_map.get(args.resolution.upper(), sl.RESOLUTION.HD720)
    p.camera_fps        = args.fps

    # Use ULTRA depth for both VO accuracy and scene analysis quality
    p.depth_mode             = sl.DEPTH_MODE.ULTRA
    p.coordinate_units       = sl.UNIT.METER
    p.coordinate_system      = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    p.depth_minimum_distance = 0.2
    p.depth_maximum_distance = 20.0

    if args.svo:
        p.set_from_svo_file(str(args.svo))
        p.svo_real_time_mode = False

    return p


def _build_tracking_params() -> sl.PositionalTrackingParameters:
    tp = sl.PositionalTrackingParameters()
    tp.enable_imu_fusion  = True
    tp.enable_area_memory = True
    tp.set_as_static      = False
    return tp


# =============================================================================
# Combined HUD overlay
# =============================================================================

def _draw_vo_hud(frame: np.ndarray, nav: NavState) -> None:
    """Burn VO navigation figures into the top-left corner of frame (in-place)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 120)
    lh    = 24

    def put(text: str, row: int, scale: float = 0.52):
        cv2.putText(frame, text, (8, 18 + row * lh),
                    font, scale, color, 1, cv2.LINE_AA)

    p, e = nav.position, nav.orientation
    put(f"Pos  X:{p[0]:+7.3f}  Y:{p[1]:+7.3f}  Z:{p[2]:+7.3f} m", 0)
    put(f"Rpy  R:{e[0]:+6.1f}  P:{e[1]:+6.1f}  Y:{e[2]:+6.1f} deg", 1)
    put(f"Speed: {nav.speed:.3f} m/s     Frame: {nav.frame_idx}", 2, 0.60)
    put(f"CumDist: {nav.cumulative_distance:.3f} m   NetDist: {nav.net_displacement:.3f} m", 3)
    if nav.imu_available:
        a, g = nav.linear_acceleration, nav.angular_velocity
        put(f"Acc  {a[0]:+.2f}  {a[1]:+.2f}  {a[2]:+.2f} m/s²", 4)
        put(f"Gyro {g[0]:+.2f}  {g[1]:+.2f}  {g[2]:+.2f} °/s", 5)
    else:
        put("IMU: not available on ZED gen-1", 4)


# =============================================================================
# Main SLAM loop
# =============================================================================

def run_slam(args: argparse.Namespace) -> None:
    scene_cfg = SceneConfig()

    # ── Open camera ──────────────────────────────────────────────────────────
    zed         = sl.Camera()
    init_params = _build_init_params(args)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"[ERROR] Cannot open ZED camera: {err}\n"
                 "Check USB/CSI connection, SDK version, and permissions.")

    cam_info = zed.get_camera_information()
    model    = str(cam_info.camera_model)
    serial   = cam_info.serial_number
    cam_w    = cam_info.camera_configuration.resolution.width
    cam_h    = cam_info.camera_configuration.resolution.height
    fps_cam  = cam_info.camera_configuration.fps

    print(f"\n{'='*64}")
    print(f"ZED Camera  : {model}  (S/N {serial})")
    print(f"Resolution  : {cam_w}×{cam_h} @ {fps_cam} FPS")
    print(f"Coord system: RIGHT_HANDED_Y_UP  |  Units: metres")
    print(f"{'='*64}\n")

    # ── Optional SVO recording ────────────────────────────────────────────────
    if args.save_svo:
        rec_params = sl.RecordingParameters(
            str(args.save_svo), sl.SVO_COMPRESSION_MODE.H264
        )
        err = zed.enable_recording(rec_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[WARN] SVO recording failed to start: {err}")

    # ── Positional tracking ───────────────────────────────────────────────────
    err = zed.enable_positional_tracking(_build_tracking_params())
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        sys.exit(f"[ERROR] Cannot enable positional tracking: {err}")

    # ── Runtime parameters ────────────────────────────────────────────────────
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold           = 50
    runtime.texture_confidence_threshold   = 100

    # ── ZED object buffers ────────────────────────────────────────────────────
    image_left = sl.Mat()
    depth_mat  = sl.Mat()
    pose       = sl.Pose()
    imu_data   = sl.SensorsData()

    # ── Navigation state ──────────────────────────────────────────────────────
    imu_available = (cam_info.camera_model != sl.MODEL.ZED)
    if not imu_available:
        print("[INFO] ZED gen-1 detected — IMU data not available.")

    tracker  = DisplacementTracker(speed_window=args.speed_window)
    vel_ema  = EMASmoothing(alpha=args.ema_alpha)
    nav      = NavState()
    nav_lock = Lock()

    # ── Live trajectory plotter ───────────────────────────────────────────────
    plotter = None
    if not args.no_plot:
        try:
            plotter = LivePlotter(view_3d=not args.view_2d)
        except Exception as e:
            print(f"[WARN] Could not start live plot: {e}  (try --no-plot)")

    # ── Console header ────────────────────────────────────────────────────────
    print(f"{'Frame':>6}  {'X':>8}  {'Y':>8}  {'Z':>8}  "
          f"{'Spd m/s':>8}  {'Yaw°':>7}  "
          f"{'LWall':>7}  {'RWall':>7}  {'Hallway':>8}  {'LOS dist':>9}")
    print("-" * 100)

    frame_idx   = 0
    prev_pos    = None
    prev_time_s = None
    traj_buffer = []

    try:
        while True:
            # ── Grab one frame ────────────────────────────────────────────────
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                if args.svo:
                    print("[INFO] SVO playback complete.")
                    break
                continue

            ts_s = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() * 1e-9
            dt   = ((ts_s - prev_time_s) if prev_time_s is not None
                    else 1.0 / max(fps_cam, 1))
            dt   = max(dt, 1e-4)

            # ── Retrieve image + depth (shared for both pipelines) ────────────
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

            frame_bgr = image_left.get_data()[:, :, :3].copy()
            depth_raw = depth_mat.get_data().copy()

            # ── VO: pose update ───────────────────────────────────────────────
            tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)

            lin_acc = np.zeros(3)
            ang_vel = np.zeros(3)

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                t    = pose.get_translation().get()
                pos  = np.array([t[0], t[1], t[2]])

                q       = pose.get_orientation().get()   # [ox, oy, oz, ow]
                ox, oy, oz, ow = q
                R = np.array([
                    [1-2*(oy**2+oz**2),   2*(ox*oy-oz*ow),   2*(ox*oz+oy*ow)],
                    [  2*(ox*oy+oz*ow), 1-2*(ox**2+oz**2),   2*(oy*oz-ox*ow)],
                    [  2*(ox*oz-oy*ow),   2*(oy*oz+ox*ow), 1-2*(ox**2+oy**2)],
                ])
                euler = rotation_matrix_to_euler(R)

                velocity = (vel_ema.update((pos - prev_pos) / dt)
                            if prev_pos is not None else np.zeros(3))
                speed_ms = float(np.linalg.norm(velocity))
                disp     = tracker.update(pos, dt)

                if imu_available and not args.no_imu:
                    zed.get_sensors_data(imu_data, sl.TIME_REFERENCE.IMAGE)
                    imu     = imu_data.get_imu_data()
                    la      = imu.get_linear_acceleration()
                    av      = imu.get_angular_velocity()
                    lin_acc = np.array([la[0], la[1], la[2]])
                    ang_vel = np.array([av[0], av[1], av[2]])

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

                traj_buffer.append(pos.copy())
                prev_pos = pos

            prev_time_s = ts_s

            # ── Scene understanding ───────────────────────────────────────────
            depth_clean = preprocess_depth(depth_raw, scene_cfg)
            walls       = detect_walls(depth_clean, scene_cfg)
            hallway     = detect_hallway(depth_clean, walls, scene_cfg)
            los_obj     = probe_line_of_sight(depth_clean, scene_cfg)
            clusters    = find_forward_clusters(depth_clean, frame_bgr, scene_cfg)

            scene = SceneState(
                walls     = walls,
                hallway   = hallway,
                los_obj   = los_obj,
                frame_idx = frame_idx,
            )

            # ── Console output (every N frames) ──────────────────────────────
            if frame_idx % args.verbose_interval == 0:
                p = nav.position
                print(
                    f"{frame_idx:>6}  "
                    f"{p[0]:>8.3f}  {p[1]:>8.3f}  {p[2]:>8.3f}  "
                    f"{nav.speed:>8.3f}  {nav.orientation[2]:>7.1f}  "
                    f"{'Y ' + f'{walls.left_dist_m:.1f}m' if walls.left_wall else 'N':>7}  "
                    f"{'Y ' + f'{walls.right_dist_m:.1f}m' if walls.right_wall else 'N':>7}  "
                    f"{'YES' if hallway.detected else 'no':>8}  "
                    f"{los_obj.dist_m:>9.2f}"
                )
                if imu_available and not args.no_imu:
                    a, g = nav.linear_acceleration, nav.angular_velocity
                    print(f"{'':>6}  IMU acc [{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}] m/s²  "
                          f"gyro [{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}] °/s")

            # ── Display ───────────────────────────────────────────────────────
            if not args.no_display:
                # Scene overlay (wall highlights, cluster boxes, scene HUD panel)
                vis = draw_scene_overlay(frame_bgr, depth_clean, scene, clusters, scene_cfg)

                # VO HUD burned into the top-left of the camera portion only
                _draw_vo_hud(vis[:frame_bgr.shape[0], :], nav)

                # Depth colourmap side-panel
                depth_col = colorise_depth(depth_clean, max_m=scene_cfg.DEPTH_MAX_M)
                pad = vis.shape[0] - depth_col.shape[0]
                if pad > 0:
                    depth_col = np.vstack([
                        depth_col,
                        np.zeros((pad, depth_col.shape[1], 3), dtype=np.uint8),
                    ])
                combined = cv2.resize(np.hstack([vis, depth_col]), None,
                                      fx=0.75, fy=0.75)

                cv2.imshow("ZED SLAM", combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[INFO] User quit.")
                    break

            # ── Live trajectory plot ──────────────────────────────────────────
            if (plotter is not None
                    and frame_idx % args.plot_interval == 0
                    and len(traj_buffer) > 1):
                traj_arr = np.array(traj_buffer)
                plotter.update(
                    traj_arr, frame_idx,
                    tracker.net_displacements[-1] if tracker.net_displacements else 0.0,
                    savgol_w=args.savgol_window,
                    savgol_p=args.savgol_poly,
                )

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

    # ── Session summary ───────────────────────────────────────────────────────
    summary = tracker.summary()
    print(f"\n{'='*64}")
    print("ZED SLAM — SESSION SUMMARY")
    print(f"{'='*64}")
    print(f"Frames processed      : {summary['total_frames']}")
    print(f"Total path length     : {summary['total_path_length']:.4f} m")
    print(f"Net displacement      : {summary['final_net_magnitude']:.4f} m")
    print(f"Mean step speed       : {summary['mean_speed']:.4f} m/frame")
    print(f"Max step speed        : {summary['max_speed']:.4f} m/frame")
    print(f"Linearity ratio       : {summary['linearity_ratio']:.3f}  (1.0 = straight)")
    print(f"{'='*64}\n")

    # ── Save trajectory ───────────────────────────────────────────────────────
    if tracker.positions:
        traj_arr = np.array(tracker.positions)
        np.savetxt("zed_trajectory.txt", traj_arr,
                   header="X(m) Y(m) Z(m)", comments="# ")
        print("Saved: zed_trajectory.txt")

    # ── Analysis plots ────────────────────────────────────────────────────────
    if not args.no_plot and len(tracker.positions) > 2:
        save_analysis_plots(
            tracker,
            savgol_w=args.savgol_window,
            savgol_p=args.savgol_poly,
            view_3d=not args.view_2d,
        )


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ZED Camera SLAM — Visual Odometry + Scene Understanding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--svo",       type=Path, default=None,
                   help="SVO file for playback (omit for live camera)")
    p.add_argument("--save-svo",  type=Path, default=None,
                   help="Record the session to an SVO file")

    # Camera
    p.add_argument("--resolution", default="HD720",
                   choices=["HD2K", "HD1080", "HD720", "VGA"],
                   help="Camera resolution mode")
    p.add_argument("--fps", type=int, default=30,
                   help="Target framerate")

    # Display / plotting
    p.add_argument("--no-display",    action="store_true",
                   help="Headless mode — no OpenCV windows")
    p.add_argument("--no-plot",       action="store_true",
                   help="Disable matplotlib (headless Jetson)")
    p.add_argument("--view-2d",       action="store_true",
                   help="2-D (X–Z) trajectory view instead of 3-D")
    p.add_argument("--plot-interval", type=int, default=10,
                   help="Update live trajectory plot every N frames")

    # VO smoothing
    p.add_argument("--ema-alpha",     type=float, default=0.7,
                   help="EMA alpha for velocity smoothing [0,1]")
    p.add_argument("--savgol-window", type=int,   default=11,
                   help="Savitzky-Golay window for post-run plots (odd)")
    p.add_argument("--savgol-poly",   type=int,   default=3,
                   help="Savitzky-Golay polynomial order")
    p.add_argument("--speed-window",  type=int,   default=10,
                   help="Rolling window size for smoothed speed estimate")

    # IMU
    p.add_argument("--no-imu", action="store_true",
                   help="Skip IMU data retrieval (ZED gen-1 compatible)")

    # Verbosity
    p.add_argument("--verbose-interval", type=int, default=10,
                   help="Print combined state every N frames")

    return p


if __name__ == "__main__":
    run_slam(_build_parser().parse_args())
