# ZED Camera SLAM — Documentation

## Overview

This system fuses two previously separate ZED camera pipelines into a unified SLAM stack:

- **Visual Odometry (VO)** — 6-DoF pose tracking, metric velocity, IMU fusion, and trajectory analysis
- **Scene Understanding** — wall/hallway detection, line-of-sight object probing, and forward-corridor cluster detection

Both pipelines share a single `sl.Camera` handle and a single `zed.grab()` call per frame, avoiding the latency and synchronization issues that arise from running two independent camera loops.

---

## File Structure

```
zed_slam_main.py      ← entry point; run this
zed_vo_core.py        ← VO data structures, tracking, smoothing, plotting
zed_scene_core.py     ← scene analysis, depth processing, visualization
```

All three files must be in the same directory. `zed_vo_core.py` has no ZED SDK calls and can be imported or unit-tested independently. `zed_scene_core.py` imports `pyzed.sl` because it also contains a standalone `run()` entry point for running scene understanding without VO; the analysis functions themselves (`preprocess_depth`, `detect_walls`, etc.) are SDK-free and are what `zed_slam_main.py` calls.

---

## Requirements

| Dependency | Notes |
|---|---|
| ZED SDK ≥ 4.x | https://www.stereolabs.com/developers/release/ |
| pyzed | Installed automatically with the ZED SDK |
| opencv-python | `pip install opencv-python` |
| numpy | `pip install numpy` |
| scipy | `pip install scipy` |
| matplotlib | `pip install matplotlib` — only needed for live/analysis plots |

Python 3.10+ recommended (uses `list[T]` type hints).

---

## Quick Start

```bash
# Live camera, all features enabled
python zed_slam_main.py

# Replay an SVO recording
python zed_slam_main.py --svo path/to/recording.svo

# Headless (no display, no plots) — suitable for Jetson without a monitor
python zed_slam_main.py --no-display --no-plot

# Record a new SVO while running live
python zed_slam_main.py --save-svo session.svo

# ZED gen-1 (no IMU)
python zed_slam_main.py --no-imu

# See all options
python zed_slam_main.py --help
```

---

## CLI Reference

### Input / Output

| Flag | Default | Description |
|---|---|---|
| `--svo PATH` | *(none)* | SVO file path for playback. Omit to use the live camera. |
| `--save-svo PATH` | *(none)* | Record the live session to an SVO file (H.264 compression). |

### Camera

| Flag | Default | Choices | Description |
|---|---|---|---|
| `--resolution MODE` | `HD720` | `HD2K`, `HD1080`, `HD720`, `VGA` | Camera resolution mode. |
| `--fps N` | `30` | — | Target framerate. Must be a rate supported by the chosen resolution. |

### Display

| Flag | Default | Description |
|---|---|---|
| `--no-display` | off | Disable all OpenCV windows. Required on headless systems. |
| `--no-plot` | off | Disable matplotlib (live trajectory window + post-run analysis plots). |
| `--view-2d` | off | Use a 2-D (X–Z top-down) trajectory view instead of 3-D. |
| `--plot-interval N` | `10` | Refresh the live trajectory plot every N frames. |

### Navigation / Smoothing

| Flag | Default | Description |
|---|---|---|
| `--ema-alpha A` | `0.7` | EMA smoothing factor for velocity. Higher = more responsive, lower = smoother. Range `[0, 1]`. |
| `--savgol-window W` | `11` | Savitzky-Golay window length for post-run trajectory plots. Must be odd. Set to `0` or `1` to disable. |
| `--savgol-poly P` | `3` | Savitzky-Golay polynomial order. Must be less than `--savgol-window`. |
| `--speed-window N` | `10` | Rolling window size for smoothed per-frame speed estimate. |

### IMU

| Flag | Default | Description |
|---|---|---|
| `--no-imu` | off | Skip IMU data retrieval. Use on ZED gen-1 cameras, which have no IMU. |

### Verbosity

| Flag | Default | Description |
|---|---|---|
| `--verbose-interval N` | `10` | Print the combined navigation + scene state row every N frames. |

---

## Console Output

On startup, the camera model, serial number, resolution, and FPS are printed. During the run, a combined row is printed every `--verbose-interval` frames:

```
 Frame         X         Y         Z   Spd m/s    Yaw°   LWall  RWall   Hallway  LOS dist
-----------------------------------------------------------------------------------------------
     0     0.000     0.000     0.000      0.000     0.0       N      N        no      inf
    10     0.031    -0.002     0.184      0.019    -0.3  Y 1.2m  Y 1.1m      YES     3.21
    20     0.055     0.001     0.371      0.021    -0.4  Y 1.2m  Y 1.1m      YES     3.18
```

On cameras with IMU (ZED 2 / 2i / X), an additional IMU line is printed underneath each row:

```
        IMU acc [+0.12 -0.03 +9.81] m/s²  gyro [+0.01 +0.00 -0.02] °/s
```

---

## Display Window

The `ZED SLAM` window shows three regions side by side at 75% scale:

```
┌──────────────────────────────────┬─────────────────┐
│  Camera image                    │  Depth colour   │
│  ├─ VO HUD (top-left)            │  map (Turbo)    │
│  ├─ Wall zone highlights         │                 │
│  └─ Cluster bounding boxes       │                 │
├──────────────────────────────────┤                 │
│  Scene HUD panel                 │  (padded)       │
└──────────────────────────────────┴─────────────────┘
```

**VO HUD** (top-left of camera image, semi-transparent background):
- Position X / Y / Z in metres
- Roll / Pitch / Yaw in degrees
- Speed in m/s and current frame index
- Cumulative path length and net displacement
- IMU accelerometer and gyroscope readings (ZED 2 / 2i / X only)

**Wall zone highlights** (semi-transparent tints):
- Orange tint over the left third → left wall detected
- Orange tint over the right third → right wall detected
- Blue tint over the centre third → front wall detected

**Cluster bounding boxes**: cyan rectangles with distance labels around depth clusters in the forward corridor.

**LOS probe rectangle**: a small rectangle at the image centre — green if an object is detected within `LOS_OBJECT_MAX_M`, grey if clear.

**Scene HUD panel** (below the camera image):
- Left / right / front wall status and distances
- Hallway detection status, estimated width, and open depth ahead
- Line-of-sight object status and distance
- Cluster count and frame index

Press **Q** or **ESC** to quit.

---

## Output Files

These files are written to the **current working directory** at the end of each session:

| File | Description |
|---|---|
| `zed_trajectory.txt` | Space-separated X Y Z positions (one row per frame). Header: `# X(m) Y(m) Z(m)`. |
| `zed_displacement_analysis.png` | 6-panel analysis figure: position over time, per-frame step size, cumulative vs net distance, linearity ratio, top-down trajectory, and height over time. |
| `zed_trajectory_plot.png` | Final smoothed trajectory — 3-D (default) or X-Z 2-D (with `--view-2d`). |

The analysis plots are generated only if `--no-plot` is not set and there are more than 2 recorded trajectory points.

---

## Architecture

```
zed_slam_main.py
│
│  One ZED camera handle
│  One grab() per frame
│
├── VO pipeline (per frame)
│   ├── zed.get_position()         → pose (translation + quaternion)
│   ├── quaternion → rotation matrix → Euler angles
│   ├── EMASmoothing               → smoothed velocity
│   ├── DisplacementTracker.update()  → cumulative/net distance, linearity
│   ├── zed.get_sensors_data()     → IMU (ZED 2/2i/X only)
│   └── NavState (thread-safe)     → current navigation snapshot
│
└── Scene pipeline (per frame)
    ├── preprocess_depth()         → NaN-masked float32 depth array
    ├── detect_walls()             → WallInfo (grid variance voting)
    ├── detect_hallway()           → HallwayInfo (parallel walls + open centre)
    ├── probe_line_of_sight()      → LOSObject (central rectangle median)
    ├── find_forward_clusters()    → [(x,y,w,h,depth), ...] (morphological blobs)
    └── SceneState                 → current scene snapshot
```

Both pipelines read from the same `depth_raw` numpy array retrieved once per frame via `zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)`.

---

## Module Reference

### `zed_vo_core.py`

#### `NavState` (dataclass)
Mutable snapshot of all navigation data for a single frame. Updated in-place each frame under `nav_lock` in the main loop.

| Field | Type | Description |
|---|---|---|
| `position` | `np.ndarray` (3,) | X, Y, Z in metres |
| `orientation` | `np.ndarray` (3,) | Roll, pitch, yaw in degrees |
| `rotation_matrix` | `np.ndarray` (3,3) | SO(3) rotation matrix |
| `velocity` | `np.ndarray` (3,) | EMA-smoothed velocity in m/s |
| `speed` | `float` | Scalar speed in m/s |
| `cumulative_distance` | `float` | Total path length in metres |
| `net_displacement` | `float` | Straight-line distance from origin in metres |
| `linearity_ratio` | `float` | `net / cumulative` — 1.0 = perfectly straight |
| `imu_available` | `bool` | False on ZED gen-1 |
| `linear_acceleration` | `np.ndarray` (3,) | m/s² |
| `angular_velocity` | `np.ndarray` (3,) | deg/s |
| `frame_idx` | `int` | Current frame number |
| `timestamp_s` | `float` | Image timestamp in seconds |

#### `DisplacementTracker`
Accumulates pose updates and computes path statistics.

```python
tracker = DisplacementTracker(speed_window=10)
stats   = tracker.update(pos, dt)   # call once per frame
summary = tracker.summary()         # call at end of session
```

`update()` returns a dict with keys: `step`, `step_mag`, `cumulative_distance`, `net_displacement`, `net_magnitude`, `smoothed_speed`, `linearity_ratio`.

`summary()` returns a dict with keys: `total_frames`, `total_path_length`, `final_net_magnitude`, `mean_speed`, `max_speed`, `linearity_ratio`.

#### `EMASmoothing`
Exponential moving average, used for velocity smoothing.

```python
ema = EMASmoothing(alpha=0.7)
smoothed = ema.update(raw_velocity)   # np.ndarray → np.ndarray
```

#### `rotation_matrix_to_euler(R)`
Converts a 3×3 rotation matrix to `[roll, pitch, yaw]` in degrees.

#### `smooth_trajectory(traj, window=11, poly=3)`
Applies Savitzky-Golay smoothing to an `(N, 3)` position array. Returns array of same shape.

#### `LivePlotter(view_3d=True)`
Real-time matplotlib trajectory window. Call `plotter.update(traj, frame_idx, net_dist)` every N frames. Call `plotter.close()` on shutdown.

#### `save_analysis_plots(tracker, savgol_w, savgol_p, view_3d)`
Generates and saves `zed_displacement_analysis.png` and `zed_trajectory_plot.png` from a completed `DisplacementTracker`.

---

### `zed_scene_core.py`

#### `SceneConfig` / `Config`
All tuneable scene analysis parameters. The class is defined as `Config` in `zed_scene_core.py`; `SceneConfig` is an alias exported for use by `zed_slam_main.py`. Edit the class attributes directly to adjust behaviour for your environment.

| Parameter | Default | Description |
|---|---|---|
| `DEPTH_MIN_M` | `0.3` | Ignore depth values closer than this (metres) |
| `DEPTH_MAX_M` | `8.0` | Ignore depth values farther than this (metres) |
| `GRID_COLS` | `12` | Horizontal cells for wall detection grid |
| `GRID_ROWS` | `8` | Vertical cells for wall detection grid |
| `WALL_VAR_THRESH` | `0.04` | Max depth variance (m²) for a cell to be "flat" |
| `WALL_MEAN_MAX_M` | `5.0` | Max mean depth for a cell to be classified as a wall |
| `HALLWAY_SIDE_MAX_M` | `2.5` | Side walls must be closer than this for a hallway call |
| `HALLWAY_CENTRE_MIN_M` | `1.2` | Centre must be at least this much farther than sides |
| `HALLWAY_OPEN_RATIO` | `1.5` | `centre_depth / side_depth` threshold |
| `LOS_WIDTH_FRAC` | `0.15` | Half-width of LOS probe rectangle (fraction of frame width) |
| `LOS_HEIGHT_FRAC` | `0.15` | Half-height of LOS probe rectangle (fraction of frame height) |
| `LOS_OBJECT_MAX_M` | `4.0` | Report a LOS object if median probe depth is below this |
| `CLUSTER_DEPTH_MAX_M` | `4.0` | Only detect clusters within this range |
| `CLUSTER_MIN_PIXELS` | `400` | Minimum connected-component size to report as a cluster |
| `CLUSTER_DILATE_ITER` | `3` | Morphological dilation iterations before clustering |
| `OVERLAY_ALPHA` | `0.45` | Transparency of wall highlight overlays |

#### `SceneState` (dataclass)
Snapshot of all scene analysis results for one frame.

| Field | Type | Description |
|---|---|---|
| `walls` | `WallInfo` | Left / right / front wall detection results |
| `hallway` | `HallwayInfo` | Hallway detection result |
| `los_obj` | `LOSObject` | Line-of-sight probe result |
| `frame_idx` | `int` | Frame number |

#### `WallInfo` (dataclass)
| Field | Type | Description |
|---|---|---|
| `left_wall` | `bool` | Left wall detected |
| `right_wall` | `bool` | Right wall detected |
| `front_wall` | `bool` | Front wall detected |
| `left_dist_m` | `float` | Mean depth of left zone (metres) |
| `right_dist_m` | `float` | Mean depth of right zone (metres) |
| `front_dist_m` | `float` | Mean depth of centre zone (metres) |

#### `HallwayInfo` (dataclass)
| Field | Type | Description |
|---|---|---|
| `detected` | `bool` | Hallway condition satisfied |
| `width_est_m` | `float` | Rough corridor width estimate (metres) |
| `centre_open_m` | `float` | Mean depth of open centre strip (metres) |

#### `LOSObject` (dataclass)
| Field | Type | Description |
|---|---|---|
| `detected` | `bool` | Object within `LOS_OBJECT_MAX_M` |
| `dist_m` | `float` | Median depth of probe region (metres) |
| `label` | `str` | Human-readable label (e.g. `"Object @ 2.41 m"`) |

#### Core functions

```python
depth_clean = preprocess_depth(depth_raw, cfg)
# Replaces invalid / out-of-range values with NaN. Returns float32 copy.

walls = detect_walls(depth_clean, cfg)
# Divides the depth image into a GRID_ROWS × GRID_COLS grid, marks flat cells
# (low variance, within range), and votes per column-third to classify walls.

hallway = detect_hallway(depth_clean, walls, cfg)
# Checks that both side walls are close, parallel, and the centre strip is
# significantly farther away. Width is estimated from horizontal FOV.

los_obj = probe_line_of_sight(depth_clean, cfg)
# Takes the median depth in a central rectangle. Returns a detected LOSObject
# if something is closer than LOS_OBJECT_MAX_M.

clusters = find_forward_clusters(depth_clean, frame_bgr, cfg)
# Morphological blob detection on the depth mask (centre 60% of frame,
# within CLUSTER_DEPTH_MAX_M). Returns a list of (x, y, w, h, mean_depth)
# tuples sorted nearest-first.

vis = draw_scene_overlay(frame_bgr, depth_clean, scene, clusters, cfg)
# Returns a new BGR image: camera frame with wall highlights, LOS rectangle,
# and cluster boxes, stacked above a scene HUD panel.

bgr = colorise_depth(depth_clean, max_m=cfg.DEPTH_MAX_M)
# Converts a float32 depth array to a COLORMAP_TURBO BGR image.
```

---

## Tuning Guide

### Wall detection is too sensitive / not sensitive enough
Adjust `WALL_VAR_THRESH` in `SceneConfig`. A lower value (e.g. `0.02`) requires surfaces to be flatter before being counted as walls. A higher value (e.g. `0.08`) allows more textured or uneven surfaces to qualify. If walls are being missed because they're far away, raise `WALL_MEAN_MAX_M`.

### Hallway detection fires in open spaces
Tighten the ratio: raise `HALLWAY_OPEN_RATIO` (e.g. to `2.0`) and lower `HALLWAY_SIDE_MAX_M` (e.g. to `1.8`). This forces the system to only call a hallway when the side walls are close and the centre is substantially more open.

### LOS probe keeps triggering on the floor
Reduce `LOS_HEIGHT_FRAC` (e.g. to `0.08`) to narrow the probe rectangle vertically, pointing it more squarely at the forward horizon rather than the ground.

### Too many / too few clusters reported
Raise `CLUSTER_MIN_PIXELS` to filter out small noise blobs, or lower it to catch smaller objects. Raise `CLUSTER_DEPTH_MAX_M` to look farther ahead.

### VO tracking drifts quickly
Try lowering `--ema-alpha` (e.g. `0.5`) for smoother velocity, and raise `--speed-window` for a longer rolling average. On challenging environments, `DEPTH_MODE.ULTRA` (already the default) gives the best pose accuracy.

### High CPU on Jetson
Use `--no-plot` to skip matplotlib entirely. Use `--no-display` if a monitor is not available. Raise `--verbose-interval` to reduce print I/O (fewer rows printed per second).

---

## Coordinate System

The ZED SDK is configured with `RIGHT_HANDED_Y_UP`, which means:

- **+X** → right
- **+Y** → up
- **+Z** → backward (camera looks toward –Z)

All positions and distances are in **metres**. The origin is the camera's position at the start of the session (or at the beginning of SVO playback).

---

## Limitations

- Wall detection uses a flat-variance heuristic and assumes surfaces are approximately fronto-parallel. Angled walls or heavily textured surfaces may not be detected reliably.
- Hallway width estimation assumes a horizontal FOV of ~90°, which is accurate for ZED 2 at HD720. Other models or resolutions may differ slightly.
- The LOS probe reports the median depth of the probe region but does not classify the object type. A future improvement would be to pass the RGB crop to a lightweight classifier.
- Cluster detection is purely geometric; overlapping clusters at similar depths may be merged into a single bounding box.
- Area memory (`enable_area_memory = True`) improves loop-closure behaviour but increases RAM usage. Disable it by editing `_build_tracking_params()` in `zed_slam_main.py` if memory is constrained.