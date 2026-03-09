# ZED VO — Visual Odometry Documentation

## Overview

`zed_vo.py` is a real-time 6-DoF navigation tracking system built for the ZED stereo camera running on an NVIDIA Jetson. It replaces a monocular visual odometry pipeline (which had no true metric scale) with ZED SDK primitives that produce physically meaningful measurements in metres.

The script tracks **where the camera is, where it's pointing, how fast it's moving, and how far it has travelled** — all in real time, frame by frame.

---

## What Is Visual Odometry?

Visual odometry (VO) is the process of estimating a camera's position and orientation over time by analysing changes between successive frames. It works similarly to how humans use visual cues to judge movement — if the scene shifts left, you moved right.

### Monocular vs Stereo VO

| | Monocular (original) | Stereo ZED (this script) |
|---|---|---|
| Scale | Guessed via pixel motion heuristic | True metric (metres) from stereo baseline |
| Drift | High — accumulates quickly | Low — IMU fusion corrects continuously |
| Depth | None | Real depth map per frame |
| IMU | None | Accelerometer + gyroscope fused in |

The core reason the ZED is better for autonomous navigation is **scale**. Monocular VO can tell you the camera moved, but not by how much in the real world. The ZED's stereo baseline (the fixed distance between its two lenses) gives the geometry needed to compute real distances.

---

## How the Pipeline Works

### 1. Camera Initialisation

```
ZED SDK opens camera → sets resolution, FPS, depth mode, coordinate system
```

The coordinate system is set to `RIGHT_HANDED_Y_UP`, meaning:
- **X** = right
- **Y** = up
- **Z** = backward (into the screen / behind camera)

Depth mode is set to `ULTRA` for maximum accuracy.

### 2. Positional Tracking

```
zed.enable_positional_tracking() → ZED runs VIO onboard
```

The ZED camera runs **Visual-Inertial Odometry (VIO)** on its own processor — it fuses optical feature tracking with IMU data internally. Every frame, `zed.get_position()` returns a full 6-DoF pose:

- **Translation** `[X, Y, Z]` in metres — where the camera is in the world
- **Orientation** as a quaternion → converted to `[roll, pitch, yaw]` in degrees

This is the key difference from the original monocular scripts, which had to manually run `cv2.findEssentialMat()` and `cv2.recoverPose()` in Python. The ZED does all of that internally, faster and more accurately.

### 3. Velocity Estimation

```
velocity = (current_position - previous_position) / dt
```

Since the ZED SDK doesn't expose velocity directly, the script computes it via finite difference on consecutive pose updates. An **Exponential Moving Average (EMA)** filter smooths out frame-to-frame jitter:

```
smoothed_velocity = alpha * raw_velocity + (1 - alpha) * previous_smoothed
```

- Higher `alpha` (→ 1.0) = trusts new data more, less smoothing
- Lower `alpha` (→ 0.0) = trusts history more, more smoothing
- Default: `0.7`

### 4. IMU Data (ZED 2 / 2i / ZED X only)

```
zed.get_sensors_data() → linear acceleration [m/s²], angular velocity [°/s]
```

The IMU provides:
- **Linear acceleration** — raw forces on the camera body (gravity + motion)
- **Angular velocity** — rotation rate around each axis

The ZED SDK already fuses IMU into the pose estimate internally. The script additionally exposes raw IMU values for external use (e.g. feeding into your own EKF).

> **Note:** The original ZED (gen-1, USB 3) has no IMU. ZED 2, 2i, and ZED X all have IMU.

### 5. Displacement Tracking (`DisplacementTracker`)

Every frame, the tracker updates these running statistics:

| Metric | Description |
|---|---|
| `step_mag` | Distance moved since last frame (metres) |
| `cumulative_distance` | Total path length travelled so far |
| `net_displacement` | Straight-line distance from the starting point |
| `smoothed_speed` | Rolling average of recent step sizes |
| `linearity_ratio` | `net / cumulative` — how straight the path is (1.0 = perfectly straight) |

### 6. Trajectory Smoothing

Two smoothing stages are applied:

**Online (per-frame):** EMA on velocity increments reduces jitter during tracking.

**Post-run:** Savitzky-Golay filter on the full recorded trajectory for clean plots. This filter fits a polynomial to a sliding window — it smooths noise while preserving the shape of peaks and turns better than a simple moving average.

---

## Output

### Console (per N frames)

```
 Frame        X        Y        Z   Spd m/s    Yaw°  Pitch°   Roll°   CumDist   NetDist
   100    0.432    0.021   -0.218     0.031    -12.4     1.2     0.3     1.243     0.489
         IMU acc [+0.12, -9.81, +0.03] m/s²   gyro [+0.02, +0.11, -0.04] °/s
```

### Live Plot

A real-time 3D (or 2D top-down) trajectory plot updates every N frames via matplotlib. The current position is shown as a red dot, start as green.

### Saved Files

| File | Contents |
|---|---|
| `zed_trajectory.txt` | Raw XYZ positions, one row per frame |
| `zed_trajectory_plot.png` | Final 3D trajectory (raw + smoothed) |
| `zed_displacement_analysis.png` | 6-panel analysis: position, step size, cumulative vs net distance, linearity, top-down path, height |

### HUD Overlay

When the camera preview window is open, navigation data is burned onto the frame in real time: position, orientation, speed, cumulative distance, and IMU readings.

---

## CLI Reference

```bash
python3 zed_vo.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--svo PATH` | None | Play back a recorded `.svo` file instead of live camera |
| `--save-svo PATH` | None | Record the session to an SVO file while running |
| `--resolution` | `HD720` | Camera resolution: `HD2K`, `HD1080`, `HD720`, `VGA` |
| `--fps` | `30` | Target framerate |
| `--no-display` | off | Disable the camera preview window (headless) |
| `--no-plot` | off | Disable all matplotlib output (headless Jetson) |
| `--view-2d` | off | Use 2D (X–Z) plot instead of 3D |
| `--plot-interval` | `10` | Update live plot every N frames |
| `--ema-alpha` | `0.7` | Velocity EMA smoothing factor |
| `--savgol-window` | `11` | Savitzky-Golay window size for post-run plots |
| `--savgol-poly` | `3` | Savitzky-Golay polynomial order |
| `--speed-window` | `10` | Rolling window size for smoothed speed |
| `--no-imu` | off | Skip IMU retrieval (use on ZED gen-1) |
| `--verbose-interval` | `10` | Print nav state every N frames |

---

## Installation & Setup

### Requirements

- NVIDIA Jetson running JetPack 5.x or 6.x
- ZED SDK 4.x or 5.x (matched to your JetPack version)
- Python 3.8–3.11

### Dependencies

```bash
# ZED Python API (run after installing ZED SDK)
python3 /usr/local/zed/get_python_api.py

# NumPy must be < 2.0 (pyzed ABI requirement)
pip3 install "numpy<2" --force-reinstall

# Other dependencies
pip3 install scipy matplotlib opencv-python

# Fix libturbojpeg if missing
sudo apt-get install libturbojpeg

# Ensure ZED libs are on the path (add to ~/.bashrc)
export LD_LIBRARY_PATH=/usr/local/zed/lib:$LD_LIBRARY_PATH
```

### Running

```bash
cd ~/Documents/rover

# Live camera
python3 zed_vo.py

# Headless (no display, no plots — for deployment)
python3 zed_vo.py --no-display --no-plot

# Replay a recording
python3 zed_vo.py --svo recording.svo

# Record while running
python3 zed_vo.py --save-svo session1.svo
```

---

## Key Classes

### `DisplacementTracker`
Accumulates pose updates and computes running navigation statistics. Call `tracker.update(position, dt)` each frame; call `tracker.summary()` at the end.

### `EMASmoothing`
Single-value exponential moving average. Used to smooth velocity computed from finite differences.

### `NavState`
Dataclass snapshot of all navigation quantities for a single frame. Thread-safe (protected by a lock) for use in multi-threaded applications.

### `LivePlotter`
Wraps matplotlib for real-time trajectory display. Disabled automatically if matplotlib is unavailable or `--no-plot` is set.

---

## Coordinate System

```
        Y (up)
        |
        |
        +---------- X (right)
       /
      /
    Z (back, toward camera)
```

All positions are in the **world frame** — fixed at the camera's location when the script started. The camera's starting pose is the origin `(0, 0, 0)`.