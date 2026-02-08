import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


# ============================
# Camera Intrinsics Utilities
# ============================

def get_camera_matrix(width, height, fov_deg=70):
    """
    Approximate camera intrinsics if calibration is unknown.
    """
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg / 2))
    cx = width / 2
    cy = height / 2
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    return K


# ============================
# Feature Tracking (KLT)
# ============================

def track_features(prev_gray, curr_gray, prev_pts, use_fb_check=True):
    """
    Track features using Lucas-Kanade optical flow + optional forward-backward check.
    
    Args:
        use_fb_check: If False, skips forward-backward check for 2x speedup
    """
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None,
        winSize=(15, 15),  # Reduced from (21, 21)
        maxLevel=2,         # Reduced from 3
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )

    if use_fb_check:
        # Forward-backward check
        prev_pts_reproj, status2, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, curr_pts, None,
            winSize=(15, 15),
            maxLevel=2
        )

        fb_error = np.linalg.norm(prev_pts - prev_pts_reproj, axis=2)
        good = (status == 1) & (status2 == 1) & (fb_error < 1.0)
    else:
        good = status == 1

    prev_pts_good = prev_pts[good.flatten()]
    curr_pts_good = curr_pts[good.flatten()]

    return prev_pts_good.reshape(-1, 2), curr_pts_good.reshape(-1, 2)


# ============================
# Pose Estimation
# ============================

def estimate_pose(pts1, pts2, K):
    """
    Estimate relative pose using Essential Matrix.
    """
    if len(pts1) < 8:
        return None, None

    E, mask = cv2.findEssentialMat(
        pts2, pts1, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        return None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, K)

    return R, t


# ============================
# Scale Heuristic (Monocular)
# ============================

def estimate_scale(prev_pts, curr_pts):
    """
    Estimate relative scale using median motion magnitude.
    (Heuristic — monocular VO has no true scale)
    """
    dists = np.linalg.norm(curr_pts - prev_pts, axis=1)
    scale = np.median(dists)
    return max(scale, 1e-3)


# ============================
# Main Monocular VO Pipeline
# ============================

def monocular_vo(video_path=None, 
                 scale_factor=1.0,
                 max_features=500,
                 min_features=200,
                 use_fb_check=True,
                 frame_skip=1,
                 plot_interval=10,
                 redetect_interval=5):
    """
    Optimized monocular visual odometry pipeline.
    
    Args:
        video_path: Path to video file (None for webcam)
        scale_factor: Resize factor for input frames (0.5 = half size)
        max_features: Maximum number of features to track
        min_features: Minimum features before re-detection
        use_fb_check: Whether to use forward-backward check (slower but more robust)
        frame_skip: Process every Nth frame (1 = all frames)
        plot_interval: Update plot every N frames
        redetect_interval: Check for re-detection every N frames
    """
    # Open video or webcam
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Failed to open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📷 Resolution: {width}x{height}, FPS: {fps}")
    print(f"⚙️  Max features: {max_features}, FB check: {use_fb_check}, Frame skip: {frame_skip}")

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read first frame.")
        return

    # Apply downscaling if requested
    if scale_factor != 1.0:
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        print(f"📐 Scaled to: {width}x{height}")

    # Camera intrinsics
    K = get_camera_matrix(width, height, fov_deg=70)
    print("📐 Camera Matrix K:\n", K)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect initial features
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )

    # Global pose
    R = np.eye(3)
    t = np.zeros((3, 1))

    # Preallocate trajectory array
    if total_frames > 0:
        trajectory = np.zeros((total_frames, 3))
    else:
        trajectory = []
    traj_idx = 0

    frame_idx = 0
    redetect_counter = 0

    # Visualization setup
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    while True:
        # Frame skipping
        for _ in range(frame_skip):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break

        # Apply downscaling
        if scale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track features
        prev_pts_tracked, curr_pts_tracked = track_features(
            prev_gray, curr_gray, prev_pts, use_fb_check=use_fb_check
        )

        if len(prev_pts_tracked) > 8:
            # Estimate pose
            R_rel, t_rel = estimate_pose(prev_pts_tracked, curr_pts_tracked, K)

            if R_rel is not None:
                # Estimate scale
                scale = estimate_scale(prev_pts_tracked, curr_pts_tracked)

                # Update global pose
                t = t + scale * (R @ t_rel)
                R = R_rel @ R

                # Store trajectory
                if isinstance(trajectory, np.ndarray):
                    trajectory[traj_idx] = t.flatten()
                else:
                    trajectory.append(t.flatten())
                traj_idx += 1

        # Re-detect features if needed (but not every frame)
        redetect_counter += 1
        if redetect_counter % redetect_interval == 0 and len(curr_pts_tracked) < min_features:
            curr_pts_tracked = cv2.goodFeaturesToTrack(
                curr_gray,
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7
            )
            print(f"🔄 Re-detected features at frame {frame_idx}")

        prev_gray = curr_gray
        prev_pts = curr_pts_tracked.reshape(-1, 1, 2)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}, Features: {len(curr_pts_tracked)}")

        # Live trajectory plot (reduced frequency)
        if frame_idx % plot_interval == 0 and traj_idx > 1:
            if isinstance(trajectory, np.ndarray):
                traj = trajectory[:traj_idx]
            else:
                traj = np.array(trajectory)
            
            ax.clear()
            ax.plot(traj[:, 0], traj[:, 2], '-b', linewidth=2)
            ax.scatter(traj[-1, 0], traj[-1, 2], c='r', s=50, zorder=5)
            ax.set_title(f"Monocular VO Trajectory (Frame {frame_idx})")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            plt.pause(0.001)

    cap.release()
    plt.ioff()

    # Trim trajectory array
    if isinstance(trajectory, np.ndarray):
        trajectory = trajectory[:traj_idx]
    else:
        trajectory = np.array(trajectory)

    # Save trajectory
    np.savetxt("trajectory_research_vo.txt", trajectory)
    print(f"✅ Saved {len(trajectory)} trajectory points to trajectory_research_vo.txt")

    # Final plot
    if len(trajectory) > 0:
        plt.figure(figsize=(10, 8))
        plt.plot(trajectory[:, 0], trajectory[:, 2], '-b', linewidth=2)
        plt.scatter(trajectory[0, 0], trajectory[0, 2], c='g', s=100, label='Start', zorder=5)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 2], c='r', s=100, label='End', zorder=5)
        plt.title("Final Monocular VO Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.savefig("trajectory_plot.png", dpi=150, bbox_inches='tight')
        print("✅ Saved trajectory plot to trajectory_plot.png")
        plt.show()


# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    """
    Usage examples:
    
    # Default (webcam, all optimizations)
    python monocular_vo_optimized.py
    
    # Video file with default settings
    python monocular_vo_optimized.py video.mp4
    
    # Maximum speed (half resolution, no FB check, skip frames)
    python monocular_vo_optimized.py video.mp4 --scale 0.5 --no-fb --skip 2
    
    # High quality (more features, FB check enabled)
    python monocular_vo_optimized.py video.mp4 --features 1000 --fb
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Monocular Visual Odometry')
    parser.add_argument('video', nargs='?', default=None, help='Video file path (default: webcam)')
    parser.add_argument('--scale', type=float, default=1.0, help='Frame scale factor (default: 1.0)')
    parser.add_argument('--features', type=int, default=500, help='Max features to track (default: 500)')
    parser.add_argument('--min-features', type=int, default=200, help='Min features before re-detect (default: 200)')
    parser.add_argument('--no-fb', action='store_true', help='Disable forward-backward check (faster)')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame (default: 1)')
    parser.add_argument('--plot-interval', type=int, default=10, help='Plot update interval (default: 10)')
    
    args = parser.parse_args()
    
    monocular_vo(
        video_path=args.video,
        scale_factor=args.scale,
        max_features=args.features,
        min_features=args.min_features,
        use_fb_check=not args.no_fb,
        frame_skip=args.skip,
        plot_interval=args.plot_interval
    )