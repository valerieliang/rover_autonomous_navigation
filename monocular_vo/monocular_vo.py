import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )

    if use_fb_check:
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
                 redetect_interval=5,
                 view_3d=True,
                 save_temporal_plots=True):

    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print(f"Max features: {max_features}, FB check: {use_fb_check}, Frame skip: {frame_skip}")
    print(f"3D visualization: {view_3d}")

    ret, frame = cap.read()
    if not ret:
        print("Cannot read first frame.")
        return

    if scale_factor != 1.0:
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        print(f"Scaled to: {width}x{height}")

    K = get_camera_matrix(width, height, fov_deg=70)
    print("Camera Matrix K:\n", K)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )

    R = np.eye(3)
    t = np.zeros((3, 1))

    trajectory = np.zeros((total_frames, 3)) if total_frames > 0 else []
    traj_idx = 0

    frame_idx = 0
    redetect_counter = 0

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    
    if view_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

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
                t = t + scale * (R @ t_rel)
                R = R_rel @ R
                trajectory[traj_idx] = t.flatten()
                traj_idx += 1

        redetect_counter += 1
        if redetect_counter % redetect_interval == 0 and len(curr_pts_tracked) < min_features:
            curr_pts_tracked = cv2.goodFeaturesToTrack(
                curr_gray,
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7
            )
            print(f"Re-detected features at frame {frame_idx}")

        prev_gray = curr_gray
        prev_pts = curr_pts_tracked.reshape(-1, 1, 2)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}, Features: {len(curr_pts_tracked)}")

        if frame_idx % plot_interval == 0 and traj_idx > 1:
            traj = trajectory[:traj_idx]
            ax.clear()
            
            if view_3d:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-b', linewidth=2)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='r', s=50, marker='o')
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', s=50, marker='o')
                
                ax.set_title(f"3D Monocular VO Trajectory (Frame {frame_idx})")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_zlabel("Z (m)")
                ax.grid(True, alpha=0.3)
                
                # Set equal aspect ratio for 3D
                max_range = np.array([
                    traj[:, 0].max() - traj[:, 0].min(),
                    traj[:, 1].max() - traj[:, 1].min(),
                    traj[:, 2].max() - traj[:, 2].min()
                ]).max() / 2.0
                
                mid_x = (traj[:, 0].max() + traj[:, 0].min()) * 0.5
                mid_y = (traj[:, 1].max() + traj[:, 1].min()) * 0.5
                mid_z = (traj[:, 2].max() + traj[:, 2].min()) * 0.5
                
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                
                # Better viewing angle
                ax.view_init(elev=20, azim=45)
            else:
                ax.plot(traj[:, 0], traj[:, 2], '-b', linewidth=2)
                ax.scatter(traj[-1, 0], traj[-1, 2], c='r', s=50)
                ax.set_title(f"Monocular VO Trajectory (Frame {frame_idx})")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Z (m)")
                ax.grid(True, alpha=0.3)
                ax.axis('equal')
            
            plt.pause(0.001)

    cap.release()
    plt.ioff()

    trajectory = trajectory[:traj_idx]
    np.savetxt("trajectory_research_vo.txt", trajectory)
    print(f"Saved {len(trajectory)} trajectory points to trajectory_research_vo.txt")

    if len(trajectory) > 0:
        # Create temporal plots
        if save_temporal_plots:
            fig_temporal = plt.figure(figsize=(15, 10))
            
            # Time axis (frame indices)
            time_axis = np.arange(len(trajectory))
            
            # Position over time
            ax1 = plt.subplot(3, 2, 1)
            ax1.plot(time_axis, trajectory[:, 0], 'r-', linewidth=1.5, label='X')
            ax1.plot(time_axis, trajectory[:, 1], 'g-', linewidth=1.5, label='Y')
            ax1.plot(time_axis, trajectory[:, 2], 'b-', linewidth=1.5, label='Z')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Position (m)')
            ax1.set_title('Position Components Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Velocity (displacement between frames)
            velocities = np.diff(trajectory, axis=0)
            vel_time = time_axis[1:]
            
            ax2 = plt.subplot(3, 2, 2)
            ax2.plot(vel_time, velocities[:, 0], 'r-', linewidth=1.5, label='Vx')
            ax2.plot(vel_time, velocities[:, 1], 'g-', linewidth=1.5, label='Vy')
            ax2.plot(vel_time, velocities[:, 2], 'b-', linewidth=1.5, label='Vz')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Velocity (m/frame)')
            ax2.set_title('Velocity Components Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Speed (magnitude of velocity)
            speed = np.linalg.norm(velocities, axis=1)
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(vel_time, speed, 'purple', linewidth=2)
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Speed (m/frame)')
            ax3.set_title('Speed Over Time')
            ax3.grid(True, alpha=0.3)
            ax3.fill_between(vel_time, speed, alpha=0.3, color='purple')
            
            # Cumulative distance
            cumulative_distance = np.zeros(len(trajectory))
            cumulative_distance[1:] = np.cumsum(speed)
            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(time_axis, cumulative_distance, 'orange', linewidth=2)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Distance (m)')
            ax4.set_title('Cumulative Distance Traveled')
            ax4.grid(True, alpha=0.3)
            ax4.fill_between(time_axis, cumulative_distance, alpha=0.3, color='orange')
            
            # Distance from origin
            distance_from_origin = np.linalg.norm(trajectory, axis=1)
            ax5 = plt.subplot(3, 2, 5)
            ax5.plot(time_axis, distance_from_origin, 'teal', linewidth=2)
            ax5.set_xlabel('Frame')
            ax5.set_ylabel('Distance (m)')
            ax5.set_title('Distance from Starting Point')
            ax5.grid(True, alpha=0.3)
            ax5.fill_between(time_axis, distance_from_origin, alpha=0.3, color='teal')
            
            # Height (Y) over time with emphasis
            ax6 = plt.subplot(3, 2, 6)
            ax6.plot(time_axis, trajectory[:, 1], 'green', linewidth=2)
            ax6.set_xlabel('Frame')
            ax6.set_ylabel('Height (m)')
            ax6.set_title('Vertical Position (Y) Over Time')
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax6.fill_between(time_axis, trajectory[:, 1], alpha=0.3, color='green')
            
            plt.tight_layout()
            plt.savefig("trajectory_temporal.png", dpi=150, bbox_inches='tight')
            print("Saved temporal analysis to trajectory_temporal.png")
            
            # Print statistics
            print("\n" + "="*50)
            print("TRAJECTORY STATISTICS")
            print("="*50)
            print(f"Total frames processed: {len(trajectory)}")
            print(f"Total distance traveled: {cumulative_distance[-1]:.2f} m")
            print(f"Final distance from origin: {distance_from_origin[-1]:.2f} m")
            print(f"Average speed: {np.mean(speed):.4f} m/frame")
            print(f"Max speed: {np.max(speed):.4f} m/frame")
            print(f"Position range - X: [{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}]")
            print(f"Position range - Y: [{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}]")
            print(f"Position range - Z: [{trajectory[:, 2].min():.2f}, {trajectory[:, 2].max():.2f}]")
            print(f"Net displacement - X: {trajectory[-1, 0]:.2f} m")
            print(f"Net displacement - Y: {trajectory[-1, 1]:.2f} m")
            print(f"Net displacement - Z: {trajectory[-1, 2]:.2f} m")
            print("="*50 + "\n")
        
        # Create final plot
        fig = plt.figure(figsize=(12, 10))
        
        if view_3d:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], '-b', linewidth=2, label='Path')
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                      c='g', s=150, marker='o', label='Start', edgecolors='black', linewidths=2)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      c='r', s=150, marker='o', label='End', edgecolors='black', linewidths=2)
            
            ax.set_title("Final 3D Monocular VO Trajectory", fontsize=14, fontweight='bold')
            ax.set_xlabel("X (m)", fontsize=12)
            ax.set_ylabel("Y (m)", fontsize=12)
            ax.set_zlabel("Z (m)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set equal aspect ratio
            max_range = np.array([
                trajectory[:, 0].max() - trajectory[:, 0].min(),
                trajectory[:, 1].max() - trajectory[:, 1].min(),
                trajectory[:, 2].max() - trajectory[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
            mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
            mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.view_init(elev=20, azim=45)
        else:
            ax = fig.add_subplot(111)
            ax.plot(trajectory[:, 0], trajectory[:, 2], '-b', linewidth=2)
            ax.scatter(trajectory[0, 0], trajectory[0, 2], c='g', s=100, label='Start')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 2], c='r', s=100, label='End')
            ax.set_title("Final Monocular VO Trajectory")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.savefig("trajectory_plot.png", dpi=150, bbox_inches='tight')
        print("Saved trajectory plot to trajectory_plot.png")
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optimized Monocular Visual Odometry with 3D Visualization')
    parser.add_argument('video', nargs='?', default=None)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--features', type=int, default=500)
    parser.add_argument('--min-features', type=int, default=200)
    parser.add_argument('--no-fb', action='store_true')
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--plot-interval', type=int, default=10)
    parser.add_argument('--2d', action='store_true', help='Use 2D visualization instead of 3D')
    parser.add_argument('--no-temporal', action='store_true', help='Skip temporal analysis plots')

    args = parser.parse_args()

    monocular_vo(
        video_path=args.video,
        scale_factor=args.scale,
        max_features=args.features,
        min_features=args.min_features,
        use_fb_check=not args.no_fb,
        frame_skip=args.skip,
        plot_interval=args.plot_interval,
        view_3d=not args.__dict__['2d'],
        save_temporal_plots=not args.no_temporal
    )