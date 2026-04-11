# ZED SLAM Pipeline

A real-time **visual odometry + scene understanding** pipeline for autonomous rovers, built on the Stereolabs ZED SDK and Python.

---

## What It Does

Each camera frame the pipeline:

- Retrieves a stereo image, depth map, and (on ZED 2/2i/X) IMU data from the ZED SDK.
- Queries on-device **6-DoF positional tracking** and derives position, orientation, velocity, path length, and net displacement.
- Runs **scene understanding** on the depth map — detecting walls, inferring hallway geometry, probing the forward line-of-sight, and finding nearby object clusters.
- Renders a combined OpenCV window (camera feed + VO HUD + scene overlay + depth colormap) and an optional live matplotlib trajectory plot.
- On exit, saves a trajectory text file and post-session analysis figures.

---

## Files

```
zed_slam_main.py      Entry point. Owns the camera handle. Runs the main loop.
zed_vo_core.py        Visual odometry helpers — pose maths, smoothing, plotting.
zed_scene_core.py     Scene understanding — wall/hallway detection, LOS probing,
                      object clustering. Also runnable as a standalone script.
```

---

## Dependencies

| Requirement | Notes |
|---|---|
| [ZED SDK](https://www.stereolabs.com/developers/release/) | Installs `pyzed` Python bindings |
| `numpy` | Array maths |
| `scipy` | Savitzky-Golay smoothing |
| `opencv-python` | Display and depth colormap |
| `matplotlib` | Live and post-session trajectory plots |

---

## Usage

```bash
# Live camera
python zed_slam_main.py

# Replay a recorded SVO file
python zed_slam_main.py --svo recording.svo

# Headless (no windows) — for Jetson without a display
python zed_slam_main.py --no-display --no-plot

# Record the session while running it
python zed_slam_main.py --save-svo output.svo

# Higher resolution, 2-D trajectory view
python zed_slam_main.py --resolution HD1080 --fps 60 --view-2d

# First-generation ZED (no IMU)
python zed_slam_main.py --no-imu
```

Press **Q** or **ESC** to stop. Run `python zed_slam_main.py --help` for all flags.

---

## Key Concepts

### Visual Odometry (`zed_vo_core.py`)

The ZED SDK's positional tracking fuses stereo visual odometry with IMU data internally. This module takes the output 4×4 pose matrix each frame, extracts position and rotation, converts rotation to roll/pitch/yaw, and computes frame-to-frame velocity via finite difference. Velocity is smoothed with an exponential moving average (EMA). Cumulative path length and net displacement are tracked over the full session by `DisplacementTracker`. The **linearity ratio** (net ÷ cumulative distance) measures how directly the rover moved — 1.0 is a perfectly straight line.

### Scene Understanding (`zed_scene_core.py`)

**Wall detection** divides the depth image into a configurable grid. Cells with low depth variance (flat surfaces) are counted per zone (left / centre / right). A zone is labelled a wall when more than 40 % of its cells are flat and close.

**Hallway detection** compares the mean depth of the left, centre, and right vertical strips. A hallway is inferred when both sides are close, the centre is significantly deeper, and the depth ratio exceeds a threshold.

**Line-of-sight probing** samples a small central rectangle. If the median finite depth is below a threshold, an object is reported directly ahead.

**Object clustering** thresholds the depth image at a maximum range, dilates the binary mask, and runs connected components to find and bound discrete nearby objects.

### Live Display

The OpenCV window shows three panels side by side:

- **Left — camera feed with overlays:** wall zone tints (blue = side wall, red = front wall), LOS probe rectangle (green = object detected), cyan bounding boxes for depth clusters, and the VO HUD in the top-left corner.
- **Bottom panel:** text readout of wall distances, hallway status, LOS distance, and cluster count.
- **Right — depth colormap:** JET-coloured depth map aligned to the camera frame height.

---

## Outputs

After the session ends, the following are saved to the working directory:

| File | Description |
|---|---|
| `zed_trajectory.txt` | Raw XYZ trajectory, one row per frame |
| `zed_displacement_analysis.png` | 6-panel figure: position, step size, cumulative vs net distance, linearity ratio, top-down path, height profile |
| `zed_trajectory_plot.png` | Full 3-D (or 2-D) trajectory, raw and Savitzky-Golay smoothed |

A session summary is printed to stdout with total frames, path length, net displacement, mean/max speed, and linearity ratio.

---

## Configuration

Scene detection parameters are all in the `SceneConfig` class in `zed_scene_core.py` and can be overridden at runtime. Key parameters:

- `DEPTH_MIN_M` / `DEPTH_MAX_M` — working depth range (default 0.3–8.0 m).
- `WALL_VAR_THRESH` — how flat a depth cell must be to count as a wall surface.
- `HALLWAY_OPEN_RATIO` — how much deeper the centre must be relative to the sides.
- `LOS_OBJECT_MAX_M` — distance threshold for forward object reporting.
- `CLUSTER_MIN_PIXELS` — minimum size of a depth cluster to report.

VO smoothing is controlled via CLI flags: `--ema-alpha`, `--savgol-window`, `--savgol-poly`, `--speed-window`.

---

## Module Boundaries

`zed_vo_core.py` and `zed_scene_core.py` never open a camera or call `pyzed.sl`. They operate entirely on NumPy arrays and plain Python dataclasses. `zed_slam_main.py` is the only file that touches `sl.Camera`. This makes the helper modules independently testable and reusable in other ZED-based projects.
