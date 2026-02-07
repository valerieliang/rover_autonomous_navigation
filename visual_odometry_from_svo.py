"""
This script performs monocular visual odometry (VO) on a ZED camera SVO recording. 
It estimates the camera’s motion over time by tracking visual features between consecutive frames and reconstructing the camera trajectory.
The code implements a basic monocular visual odometry pipeline using ZED SVO data and OpenCV, producing an estimated camera trajectory purely from image motion analysis.
"""

import sys
import numpy as np
import cv2
import pyzed.sl as sl


def main(svo_path):
    zed = sl.Camera()

    # Configure ZED to read from an SVO file instead of a live camera
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False  # play SVO as fast as possible
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # we only need RGB images for VO

    # Open the SVO file
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open SVO:", err)
        exit(1)

    runtime_params = sl.RuntimeParameters()

    image = sl.Mat()

    # Extract camera intrinsics (needed for essential matrix computation)
    cam_info = zed.get_camera_information()
    fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
    fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
    cx = cam_info.camera_configuration.calibration_parameters.left_cam.cx
    cy = cam_info.camera_configuration.calibration_parameters.left_cam.cy

    # Construct the camera intrinsic matrix K
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    # Initialize ORB feature detector (fast + robust for VO)
    orb = cv2.ORB_create(2000)

    # Brute-force matcher for ORB binary descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_gray = None   # previous grayscale frame
    prev_kp = None     # previous keypoints
    prev_des = None    # previous descriptors

    # Initialize global pose: rotation R and translation t
    R = np.eye(3)              # world-to-camera rotation
    t = np.zeros((3, 1))       # world-to-camera translation

    trajectory = []  # store camera positions over time

    frame_idx = 0

    while True:
        # Grab the next frame from the SVO
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            break  # end of SVO file

        # Retrieve the left camera image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()

        # Convert RGB image to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and compute descriptors in the current frame
        kp, des = orb.detectAndCompute(gray, None)

        # Only compute motion if we have a previous frame and valid descriptors
        if prev_gray is not None and des is not None and prev_des is not None:

            # Match features between previous and current frame
            matches = bf.match(prev_des, des)

            # Sort matches by descriptor distance (smaller = better match)
            matches = sorted(matches, key=lambda x: x.distance)

            # Need enough correspondences to estimate motion
            if len(matches) > 8:

                # Extract matched 2D points from both frames
                pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

                # Estimate the essential matrix using RANSAC (robust to outliers)
                E, mask = cv2.findEssentialMat(
                    pts_curr, pts_prev, K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                if E is not None:
                    # Recover relative rotation and translation between frames
                    _, R_rel, t_rel, mask = cv2.recoverPose(E, pts_curr, pts_prev, K)

                    # Update global translation:
                    # t_new = t_old + R_old * t_rel
                    t = t + R @ t_rel

                    # Update global rotation:
                    # R_new = R_rel * R_old
                    R = R_rel @ R

                    # Store the camera position in world coordinates
                    trajectory.append(t.flatten())

        # Save current frame data for next iteration
        prev_gray = gray
        prev_kp = kp
        prev_des = des

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")

    zed.close()

    trajectory = np.array(trajectory)
    print("Trajectory shape:", trajectory.shape)

    # Save trajectory to disk for later visualization / analysis
    np.savetxt("trajectory.txt", trajectory)
    print("Saved trajectory to trajectory.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    main(sys.argv[1])
