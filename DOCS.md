# ZED SLAM Pipeline — Full Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Dependencies](#installation--dependencies)
4. [Quick Start](#quick-start)
5. [CLI Reference — `zed_slam_main.py`](#cli-reference--zed_slam_mainpy)
6. [Module Reference](#module-reference)
   - [zed_slam_main.py](#zed_slam_mainpy)
   - [zed_vo_core.py](#zed_vo_corepy)
   - [zed_scene_core.py](#zed_scene_corepy)
7. [Data Structures](#data-structures)
8. [Configuration & Tuning](#configuration--tuning)
9. [Outputs & Saved Files](#outputs--saved-files)
10. [Standalone Scene Understanding](#standalone-scene-understanding)
11. [Platform Notes (Jetson / Headless)](#platform-notes-jetson--headless)

---

## Overview

This pipeline fuses **visual odometry (VO)** and **scene understanding** into a single, real-time loop driven by a Stereolabs ZED stereo camera. It is designed for autonomous rover platforms and runs on any host that supports the ZED SDK, including NVIDIA Jetson devices.

**What the pipeline does per frame:**

- Grabs a stereo image pair + depth map from the ZED camera (or an SVO replay file).
- Queries the ZED SDK's on-device positional tracking for a 6-DoF pose estimate with optional IMU fusion.
- Derives velocity, cumulative path length, net displacement, and linearity ratio from successive poses.
- Analyses the depth map to detect walls, infer hallway geometry, probe the forward line-of-sight, and find nearby object clusters.
- Burns a combined HUD onto the live OpenCV window (VO figures top-left, scene status bottom panel, depth colormap side panel).
- Optionally streams a live matplotlib trajectory window and saves post-session analysis figures.

---

## Architecture

```
zed_slam_main.py          ← single entry point, owns sl.Camera handle
│
├── pyzed.sl              ← ZED SDK Python bindings
│
├── zed_vo_core.py        ← all VO logic (no camera access)
│   ├── NavState          dataclass — per-frame navigation snapshot
│   ├── DisplacementTracker — accumulates path statistics
│   ├── EMASmoothing      — velocity smoothing
│   ├── rotation_matrix_to_euler
│   ├── smooth_trajectory (Savitzky-Golay)
│   ├── LivePlotter       — real-time matplotlib window
│   └── save_analysis_plots — post-session figure suite
│
└── zed_scene_core.py     ← all scene-understanding logic (no camera access)
    ├── SceneConfig / Config  — tuneable parameter class
    ├── WallInfo, HallwayInfo, LOSObject, SceneState  — result dataclasses
    ├── preprocess_depth  — NaN-fill, range clamp
    ├── detect_walls      — grid-cell flatness voting
    ├── detect_hallway    — corridor geometry inference
    ├── probe_line_of_sight — centre-patch median probe
    ├── find_forward_clusters — connected-component object detection
    ├── draw_overlay / draw_scene_overlay — OpenCV HUD rendering
    └── colorise_depth    — JET depth colormap panel
```

The camera handle is opened **once** in `zed_slam_main.py`. Both sub-modules receive plain NumPy arrays and dataclasses — they never call `sl.Camera` themselves. This separation means each module can be unit-tested or imported independently.

---

## Installation & Dependencies

### ZED SDK

Download and install from [stereolabs.com/developers/release](https://www.stereolabs.com/developers/release/). The SDK version must match the camera firmware and the `pyzed` Python bindings.

### Python packages

```
pip install numpy scipy opencv-python matplotlib
```

`pyzed` is installed automatically by the ZED SDK installer. Do **not** install it separately from PyPI.

### File layout

All three Python files must be in the same directory:

```
your_project/
├── zed_slam_main.py
├── zed_vo_core.py
└── zed_scene_core.py
```

---

## Quick Start

```bash
# Live camera, all features enabled
python zed_slam_main.py

# Replay a previously recorded SVO file
python zed_slam_main.py --svo path/to/recording.svo

# Headless (no GUI windows) — suitable for Jetson without a display
python zed_slam_main.py --no-display --no-plot

# Record a session while running it
python zed_slam_main.py --save-svo output.svo

# Use HD1080 at 60 FPS with a 2-D (top-down) trajectory view
python zed_slam_main.py --resolution HD1080 --fps 60 --view-2d
```

Press **Q** or **ESC** in any OpenCV window to stop the run cleanly.

---

## CLI Reference — `zed_slam_main.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--svo PATH` | Path | `None` | SVO file for replay. Omit to use the live camera. |
| `--save-svo PATH` | Path | `None` | Record the live session to an SVO file while running. |
| `--resolution` | choice | `HD720` | Camera resolution. One of `HD2K`, `HD1080`, `HD720`, `VGA`. |
| `--fps` | int | `30` | Target framerate. |
| `--no-display` | flag | off | Disable all OpenCV windows (headless mode). |
| `--no-plot` | flag | off | Disable the matplotlib live trajectory window. |
| `--view-2d` | flag | off | Use a top-down (X–Z) 2-D trajectory view instead of 3-D. |
| `--plot-interval N` | int | `10` | Refresh the live trajectory plot every N frames. |
| `--ema-alpha α` | float | `0.7` | EMA smoothing factor for velocity. Range `[0, 1]`. Higher = more responsive. |
| `--savgol-window W` | int | `11` | Savitzky-Golay window size for post-run plots. Must be odd. |
| `--savgol-poly P` | int | `3` | Savitzky-Golay polynomial order. Must be less than `--savgol-window`. |
| `--speed-window N` | int | `10` | Rolling window size for the smoothed speed estimate. |
| `--no-imu` | flag | off | Skip IMU retrieval. Required for first-generation ZED cameras. |
| `--verbose-interval N` | int | `10` | Print combined state to stdout every N frames. |

---

## Module Reference

### `zed_slam_main.py`

The entry point. Owns the `sl.Camera` handle from open to close.

#### `run_slam(args)`

The main loop. Performs in order each frame:

1. `zed.grab(runtime)` — acquire frame.
2. Retrieve left image, depth map, and (optionally) IMU sensors data.
3. Query `zed.get_position()` for the current 6-DoF pose.
4. Extract the 3×3 rotation matrix, convert to Euler angles via `rotation_matrix_to_euler`.
5. Compute frame-to-frame velocity via finite difference; smooth with `EMASmoothing`.
6. Update `DisplacementTracker` and pack all quantities into a `NavState` snapshot (under a `threading.Lock`).
7. Run scene understanding: `preprocess_depth` → `detect_walls` → `detect_hallway` → `probe_line_of_sight` → `find_forward_clusters`.
8. If display is active, compose the combined window: scene overlay + VO HUD + depth colormap side panel.
9. Every `--plot-interval` frames, refresh the live matplotlib trajectory.

On exit (KeyboardInterrupt or Q keypress): disables positional tracking, disables recording, closes the camera, prints the session summary, saves `zed_trajectory.txt`, and (unless `--no-plot`) saves the analysis PNG figures.

#### `_build_init_params(args)`

Maps CLI arguments to an `sl.InitParameters` object. Always sets depth mode to `ULTRA` and coordinate system to `RIGHT_HANDED_Y_UP` with metric units.

#### `_build_tracking_params()`

Returns an `sl.PositionalTrackingParameters` with IMU fusion and area memory enabled.

#### `_draw_vo_hud(frame, nav)`

Renders position (XYZ), orientation (roll/pitch/yaw), speed, cumulative distance, net displacement, and (if available) IMU acceleration and gyro into a semi-transparent box in the top-left of `frame`. Modifies the array in-place.

---

### `zed_vo_core.py`

Pure-Python VO helpers. No ZED SDK calls. Importable and testable independently.

#### `NavState`

A frozen-per-frame dataclass carrying every navigation quantity for one grab cycle:

| Field | Type | Description |
|---|---|---|
| `position` | `np.ndarray (3,)` | XYZ position in metres |
| `orientation` | `np.ndarray (3,)` | Roll, pitch, yaw in degrees |
| `rotation_matrix` | `np.ndarray (3,3)` | SO(3) rotation from ZED pose |
| `velocity` | `np.ndarray (3,)` | Finite-difference velocity in m/s |
| `speed` | `float` | Scalar speed in m/s |
| `cumulative_distance` | `float` | Total path length in metres |
| `net_displacement` | `float` | Straight-line distance from origin |
| `linearity_ratio` | `float` | `net / cumulative` — 1.0 = perfectly straight |
| `imu_available` | `bool` | True for ZED 2 / 2i / X |
| `linear_acceleration` | `np.ndarray (3,)` | m/s² from IMU |
| `angular_velocity` | `np.ndarray (3,)` | deg/s from IMU |
| `frame_idx` | `int` | Frame counter |
| `timestamp_s` | `float` | Camera timestamp in seconds |

#### `DisplacementTracker`

Accumulates pose updates and exposes navigation statistics.

**Constructor:** `DisplacementTracker(speed_window=10)`

**`update(pos, dt) → dict`**

Call once per frame. Returns:

| Key | Description |
|---|---|
| `step` | 3-vector displacement from previous frame |
| `step_mag` | Scalar magnitude of `step` |
| `cumulative_distance` | Total path length so far |
| `net_displacement` | 3-vector from origin |
| `net_magnitude` | Scalar net displacement |
| `smoothed_speed` | Rolling-window mean step / `dt` |
| `linearity_ratio` | `net_magnitude / cumulative_distance` |

**`summary() → dict`**

Returns aggregate statistics over the whole session: `total_frames`, `total_path_length`, `final_net_magnitude`, `mean_speed`, `max_speed`, `linearity_ratio`.

#### `EMASmoothing`

Exponential Moving Average filter for `np.ndarray` values.

```python
ema = EMASmoothing(alpha=0.7)
smoothed_velocity = ema.update(raw_velocity)
```

Higher `alpha` tracks the signal more closely; lower `alpha` smooths more aggressively.

#### `smooth_trajectory(traj, window=11, poly=3) → np.ndarray`

Applies a Savitzky-Golay filter independently to each of the X, Y, Z columns of an `(N, 3)` trajectory array. Window length is automatically clamped to the array length if needed.

#### `rotation_matrix_to_euler(R) → np.ndarray`

Converts a 3×3 SO(3) rotation matrix to `[roll, pitch, yaw]` in degrees using the standard ZYX convention. Handles the gimbal-lock singularity (`sy < 1e-6`).

#### `LivePlotter`

Real-time matplotlib trajectory window, updated every N frames.

**Constructor:** `LivePlotter(view_3d=True)`

**`update(traj, frame_idx, net_dist, savgol_w, savgol_p)`**

Clears and redraws the axes with the current trajectory. Applies Savitzky-Golay smoothing before drawing if `savgol_w > 1`. Start point is green, current position is red.

**`close()`**

Turns off interactive mode. Call once at the end of the session.

#### `save_analysis_plots(tracker, savgol_w, savgol_p, view_3d)`

Produces two PNG files at the end of a session:

- **`zed_displacement_analysis.png`** — 2×3 subplot figure: position over time, per-frame step size, cumulative vs. net distance, linearity ratio, XZ top-down trajectory, and height (Y) over time.
- **`zed_trajectory_plot.png`** — Full-resolution 3-D (or 2-D) trajectory with raw and smoothed overlays.

---

### `zed_scene_core.py`

Scene understanding functions. No ZED SDK calls after data retrieval. Can also be run as a **standalone script** (see [Standalone Scene Understanding](#standalone-scene-understanding)).

#### `preprocess_depth(depth_np, cfg) → np.ndarray`

Sanitises a raw `float32` depth array from `sl.Mat.get_data()`:
- Replaces all non-finite values with `NaN`.
- Replaces values outside `[cfg.DEPTH_MIN_M, cfg.DEPTH_MAX_M]` with `NaN`.

Returns a cleaned `float32` array of the same shape.

#### `detect_walls(depth, cfg) → WallInfo`

Divides the depth image into a `GRID_ROWS × GRID_COLS` grid. For each cell, computes the mean and variance of finite depth values. A cell is marked **flat** if its variance is below `WALL_VAR_THRESH` and its mean depth is below `WALL_MEAN_MAX_M`.

The grid is then split into left, centre, and right column thirds. A zone is classified as a wall if more than 40 % of its cells are flat. Returns a `WallInfo` with boolean flags and mean distances for each zone.

#### `detect_hallway(depth, walls, cfg) → HallwayInfo`

Computes the mean depth of the left, centre, and right vertical strips of the image. Declares a hallway when all of the following hold:

1. Both side strips have mean depth below `HALLWAY_SIDE_MAX_M`.
2. The centre strip mean depth minus the average side depth exceeds `HALLWAY_CENTRE_MIN_M`.
3. The ratio of centre depth to average side depth exceeds `HALLWAY_OPEN_RATIO`.

Returns a `HallwayInfo` with an estimated corridor width (left + right mean depth) and the centre open depth.

#### `probe_line_of_sight(depth, cfg) → LOSObject`

Samples depth values inside a central rectangle of fractional size `LOS_WIDTH_FRAC × LOS_HEIGHT_FRAC`. Reports an object if the median finite depth is below `LOS_OBJECT_MAX_M`. The `label` field is currently empty and intended as a hook for a downstream classifier.

#### `find_forward_clusters(depth, frame_bgr, cfg) → list[tuple]`

Thresholds the depth image at `CLUSTER_DEPTH_MAX_M`, dilates the binary mask, and runs OpenCV connected components. Returns a list of `(x, y, w, h, mean_depth)` tuples for each cluster larger than `CLUSTER_MIN_PIXELS`.

#### `draw_overlay(frame, depth, scene, clusters, cfg) → np.ndarray`

Composes the scene visualisation. Renders:
- Semi-transparent blue tint over detected left/right wall zones.
- Red tint over a detected front wall.
- Green (object detected) or grey (clear) rectangle for the LOS probe region.
- Cyan bounding boxes with distance labels for each detected cluster.
- A dark bottom panel showing wall distances, hallway status, LOS status, and cluster count.

Returns a new array (`frame` is not modified).

#### `colorise_depth(depth, max_m) → np.ndarray`

Normalises the cleaned depth array to `[0, 255]` and applies OpenCV's `COLORMAP_JET`. Returns a BGR `uint8` array of the same height and width as `depth`, suitable for side-by-side display.

#### `SceneConfig`

Public alias for `Config`. Use `SceneConfig` when importing from `zed_slam_main.py`. All parameters are class-level attributes and can be overridden at runtime:

```python
from zed_scene_core import SceneConfig
cfg = SceneConfig()
cfg.LOS_OBJECT_MAX_M = 2.5   # tighten line-of-sight detection threshold
```

---

## Data Structures

### `WallInfo`

| Field | Type | Description |
|---|---|---|
| `left_wall` | `bool` | Left zone classified as a wall |
| `right_wall` | `bool` | Right zone classified as a wall |
| `front_wall` | `bool` | Centre zone classified as a wall |
| `left_dist_m` | `float` | Mean depth of left zone in metres |
| `right_dist_m` | `float` | Mean depth of right zone in metres |
| `front_dist_m` | `float` | Mean depth of centre zone in metres |

### `HallwayInfo`

| Field | Type | Description |
|---|---|---|
| `detected` | `bool` | True when hallway criteria are all met |
| `width_est_m` | `float` | Rough width estimate (left + right mean depth) in metres |
| `centre_open_m` | `float` | Mean depth of the open centre strip |

### `LOSObject`

| Field | Type | Description |
|---|---|---|
| `detected` | `bool` | True when median probe depth < `LOS_OBJECT_MAX_M` |
| `dist_m` | `float` | Median depth in metres of the probe region |
| `label` | `str` | Object label — empty string (classifier hook) |

### `SceneState`

Aggregates one frame's worth of scene analysis:

```python
@dataclass
class SceneState:
    walls:     WallInfo
    hallway:   HallwayInfo
    los_obj:   LOSObject
    frame_idx: int
```

---

## Configuration & Tuning

All scene parameters live in `SceneConfig` (`zed_scene_core.py`). The table below describes each one and suggests directions for adjustment.

| Parameter | Default | Effect |
|---|---|---|
| `DEPTH_MIN_M` | `0.3` | Ignore depth closer than this. Increase to reduce noise from camera housing. |
| `DEPTH_MAX_M` | `8.0` | Ignore depth farther than this. Reduce for indoor-only use. |
| `GRID_COLS` / `GRID_ROWS` | `12` / `8` | Grid resolution for wall detection. Finer grids catch smaller surfaces. |
| `WALL_VAR_THRESH` | `0.04` | Max depth variance (m²) for a cell to count as flat. Increase in textured environments. |
| `WALL_MEAN_MAX_M` | `5.0` | Far walls (beyond this) are not classified. |
| `HALLWAY_SIDE_MAX_M` | `2.5` | Side walls must be closer than this. Increase for wider corridors. |
| `HALLWAY_CENTRE_MIN_M` | `1.2` | Minimum depth difference between centre and sides. |
| `HALLWAY_OPEN_RATIO` | `1.5` | Centre-to-side depth ratio. Reduce to detect narrower hallways. |
| `LOS_WIDTH_FRAC` / `LOS_HEIGHT_FRAC` | `0.15` | Fractional half-size of the probe rectangle. |
| `LOS_OBJECT_MAX_M` | `4.0` | Report an LOS object if median probe depth is below this. |
| `CLUSTER_DEPTH_MAX_M` | `4.0` | Only search for clusters within this range. |
| `CLUSTER_MIN_PIXELS` | `400` | Minimum cluster area in pixels. Increase to suppress noise clusters. |
| `CLUSTER_DILATE_ITER` | `3` | Morphological dilation passes before clustering. |
| `OVERLAY_ALPHA` | `0.45` | Opacity of wall highlight overlays. |

VO smoothing is controlled via CLI flags (`--ema-alpha`, `--savgol-window`, `--savgol-poly`, `--speed-window`).

---

## Outputs & Saved Files

All output files are written to the **working directory** when the session ends.

| File | Condition | Description |
|---|---|---|
| `zed_trajectory.txt` | Always (if frames > 0) | Plain-text `X Y Z` trajectory, one row per frame. Header line starts with `#`. |
| `zed_displacement_analysis.png` | Unless `--no-plot` and frames > 2 | 6-panel analysis figure (see `save_analysis_plots`). |
| `zed_trajectory_plot.png` | Unless `--no-plot` and frames > 2 | Full-resolution 3-D or 2-D trajectory figure. |
| `<name>.svo` | Only if `--save-svo` was used | H.264-compressed SVO recording for later replay. |

The session summary is always printed to stdout:

```
================================================================
ZED SLAM — SESSION SUMMARY
================================================================
Frames processed      : 1420
Total path length     : 18.3421 m
Net displacement      : 12.0034 m
Mean step speed       : 0.0129 m/frame
Max step speed        : 0.0471 m/frame
Linearity ratio       : 0.654  (1.0 = straight)
================================================================
```

---

## Standalone Scene Understanding

`zed_scene_core.py` can be run directly without the VO pipeline:

```bash
python zed_scene_core.py                    # live camera
python zed_scene_core.py --svo file.svo     # SVO replay
python zed_scene_core.py --no-display       # headless
```

This mode runs the same `detect_walls` / `detect_hallway` / `probe_line_of_sight` / `find_forward_clusters` loop and renders the same OpenCV overlay window, but does not perform positional tracking, velocity estimation, IMU readout, or trajectory recording.

It is useful for tuning the `Config` parameters before integrating the full pipeline, or for running on hardware where positional tracking is handled by a separate node.

---

## Platform Notes (Jetson / Headless)

**NVIDIA Jetson (JetPack 5 / 6):**

The ZED SDK provides Jetson-specific installers. Use `--no-plot` to skip matplotlib, which requires a display server. Use `--no-display` for fully headless operation. Recommended minimum spec: Jetson Orin Nano with ZED 2i at `HD720` 30 FPS.

**Headless Linux (no display server):**

```bash
python zed_slam_main.py --no-display --no-plot
```

Trajectory and analysis files are still saved. Console output at `--verbose-interval` is the primary monitoring channel.

**SVO file replay:**

SVO replay disables `svo_real_time_mode`, so frames are processed as fast as the host can manage rather than at the original capture rate. This is intentional for offline analysis. Use a live camera if real-time timing is required.

**ZED gen-1 (original ZED, no IMU):**

Pass `--no-imu` to suppress the `get_sensors_data` call. The VO pipeline and scene understanding operate identically; the IMU rows in the HUD will display `"IMU: not available on ZED gen-1"`.
