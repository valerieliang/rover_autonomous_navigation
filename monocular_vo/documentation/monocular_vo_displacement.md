# monocular_vo_displacement.py

An extension of `monocular_vo_smoothed.py` that computes camera displacement metrics in real time — on every frame, as poses are accumulated — rather than as a post-processing step.

---

## What's New

The smoothed pipeline already tracked position, but displacement had to be derived offline from the saved trajectory file. This script introduces a `DisplacementTracker` class that runs inside the main loop and produces a full set of displacement metrics the moment each new pose is available, with no buffering or post-processing required.

### `DisplacementTracker` class

Instantiated once before the loop and updated on every frame with the current position vector `t`:

```python
disp = tracker.update(t.flatten())
```

Each call returns a dictionary with the following fields:

| Field | Description |
|---|---|
| `frame_displacement` | 3D vector `(ΔX, ΔY, ΔZ)` — step from the previous pose |
| `frame_magnitude` | Euclidean length of that step |
| `cumulative_distance` | Running sum of all step magnitudes (total path length so far) |
| `net_displacement` | 3D vector from the origin to the current position |
| `net_magnitude` | Straight-line distance from the starting point |
| `smoothed_speed` | Rolling mean of the last N step magnitudes |
| `linearity_ratio` | `net_magnitude / cumulative_distance` |

The **linearity ratio** deserves particular attention. A value of `1.0` means the camera has moved in a perfectly straight line. Values approaching `0` indicate the camera has looped back near its origin, or that drift is causing the estimated path to wander significantly from the true trajectory.

### Real-time console output

A summary row is printed every N frames (controlled by `--verbose`):

```
 Frame      Step   CumDist   NetDist  Speed(avg)  Linearity  NetVec (X,Y,Z)
------------------------------------------------------------------------------------------
     0    0.0000    0.0000    0.0000      0.0000      1.000  (+0.000, +0.000, +0.000)
    10    2.1433   21.0442   18.3201      2.1044      0.871  (+12.310, -2.104, +13.812)
    20    1.8901   39.7230   22.4411      1.9862      0.565  (+15.231, -3.201, +16.442)
```

### Live plot title

The trajectory preview title updates each refresh to show the current net displacement, giving an at-a-glance sense of how far the camera has moved without looking at the console.

---

## Requirements

```
opencv-python
numpy
matplotlib
scipy
```

Install with:

```bash
pip install opencv-python numpy matplotlib scipy
```

---

## Usage

```bash
python monocular_vo_displacement.py [video] [options]
```

Omitting `video` opens the default webcam (device 0).

### Arguments

| Argument | Default | Description |
|---|---|---|
| `video` | webcam | Path to input video file |
| `--scale` | `1.0` | Resize factor applied to each frame |
| `--features` | `500` | Max corners for Shi-Tomasi detection |
| `--min-features` | `200` | Threshold below which features are re-detected |
| `--no-fb` | off | Disable forward-backward consistency check in KLT |
| `--skip` | `1` | Process every Nth frame |
| `--plot-interval` | `10` | Refresh the live trajectory plot every N frames |
| `--2d` | off | Use a 2D (X-Z) plot instead of 3D |
| `--ema-alpha` | `0.7` | EMA smoothing factor `[0, 1]`. Higher = less smoothing |
| `--savgol-window` | `15` | Savitzky-Golay window (odd integer; 0 or 1 = off) |
| `--savgol-poly` | `3` | Savitzky-Golay polynomial order |
| `--speed-window` | `10` | Rolling window size for real-time speed estimate |
| `--verbose` | `10` | Print displacement stats every N frames |

### Examples

```bash
# Run on a video file with defaults
python monocular_vo_displacement.py my_drive.mp4

# Print displacement every frame, heavy smoothing
python monocular_vo_displacement.py my_drive.mp4 --verbose 1 --ema-alpha 0.4 --savgol-window 31

# Webcam, 2D view, wider speed window
python monocular_vo_displacement.py --2d --speed-window 20

# Fast pass: downscale, skip frames, no FB check
python monocular_vo_displacement.py long_video.mp4 --scale 0.5 --skip 2 --no-fb
```

---

## Outputs

| File | Description |
|---|---|
| `trajectory_displacement.txt` | Raw XYZ trajectory, one point per line |
| `displacement_analysis.png` | 6-panel displacement analysis figure (see below) |
| `trajectory_plot.png` | Final trajectory, raw (faint) + smoothed (bold) |

### `displacement_analysis.png` panels

| Panel | Content |
|---|---|
| Top-left | Per-frame displacement magnitude (step size over time) |
| Top-center | Per-frame displacement components ΔX, ΔY, ΔZ |
| Top-right | Cumulative path length vs net displacement, with "wasted motion" shaded between them |
| Bottom-left | Linearity ratio over time |
| Bottom-center | Raw speed with rolling average overlaid |
| Bottom-right | Top-down (X-Z) trajectory, raw and smoothed |

---

## How the Pipeline Works

```
Frame N
  │
  ├─ KLT optical flow tracking (+ optional F-B check)
  │
  ├─ Essential Matrix → R, t (RANSAC)
  │
  ├─ Heuristic scale (median optical flow magnitude)
  │
  ├─ EMA smoothing on translation increment
  │
  ├─ Accumulate pose: t += smoothed_increment, R = R_rel @ R
  │
  ├─ DisplacementTracker.update(t)          ◄── NEW
  │     ├─ frame_displacement  = t − t_prev
  │     ├─ cumulative_distance += |frame_displacement|
  │     ├─ net_displacement    = t − origin
  │     ├─ smoothed_speed      = rolling mean of recent steps
  │     └─ linearity_ratio     = net / cumulative
  │
  └─ Print to console (every --verbose frames)

End of video
  │
  ├─ Print summary statistics
  ├─ Savitzky-Golay smoothing on full trajectory
  └─ Save displacement_analysis.png + trajectory_plot.png
```

---

## Understanding the Metrics

**Cumulative distance vs net displacement.** These two will always diverge over time unless the camera moves in a perfectly straight line. A large gap between them means one of three things: the camera changed direction frequently, the path was curved, or drift is accumulating in the pose estimate. The shaded "wasted motion" region in the top-right panel makes this gap visible at a glance.

**Linearity ratio.** Useful as a quick drift indicator. In a well-behaved sequence with genuine forward motion, the ratio should stay above `0.7` or so. If it collapses toward `0` early in the sequence, drift or degenerate motion is likely the cause rather than genuine looping.

**Smoothed speed.** The rolling average (`--speed-window`) removes single-frame spikes caused by tracking failures or sudden scale estimate jumps. Widening the window gives a more stable reading at the cost of slower response to genuine speed changes.

---

## Limitations

All displacement values are in **optical flow units**, not real-world meters. The heuristic scale factor makes values comparable within a single run but not across different videos or sessions. To obtain metric displacement, a known reference distance in the scene is required to calibrate the scale constant.

Monocular VO drift means displacement estimates become less reliable over longer sequences. The linearity ratio and the cumulative vs net distance gap both serve as indirect indicators of how much drift may be present, but they cannot correct for it.