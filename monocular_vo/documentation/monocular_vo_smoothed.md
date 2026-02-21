# monocular_vo_smoothed.py

An extension of the original `monocular_vo.py` pipeline with two layers of trajectory smoothing to reduce noise and jitter in the estimated camera path.

---

## What's New

The original script accumulates pose increments frame-by-frame using raw Essential Matrix decomposition. Because monocular visual odometry has no ground-truth scale and relies on heuristic scale estimation (median optical flow magnitude), the resulting trajectory tends to be noisy — especially under fast motion, lighting changes, or when feature tracking degrades. This version adds two complementary smoothing passes.

### 1. Exponential Moving Average (EMA) — online, per-frame

An `EMASmoothing` class wraps each translation increment before it is added to the running pose:

```
smoothed_increment = alpha * raw_increment + (1 - alpha) * previous_smoothed
```

This runs in real time on every frame. It damps sudden spikes in the estimated motion without introducing a large buffer or delay, making the live trajectory preview noticeably more stable.

**Parameter:** `--ema-alpha` (default: `0.7`)

| Value | Effect |
|-------|--------|
| `1.0` | No smoothing — raw increments pass through unchanged |
| `0.7` | Default — light smoothing, preserves responsiveness |
| `0.5` | Moderate smoothing |
| `0.3` | Heavy smoothing — may lag behind sharp turns |

### 2. Savitzky-Golay Filter — offline, on the completed trajectory

After processing all frames, `smooth_trajectory()` applies `scipy.signal.savgol_filter` independently along each axis (X, Y, Z). Unlike a plain moving average, the Savitzky-Golay filter fits a polynomial to each window, which better preserves peaks, curves, and overall trajectory shape while removing high-frequency noise.

This smoothed version is used for all final plots and statistics. The raw trajectory is still shown faintly in the background so you can compare.

**Parameters:** `--savgol-window` (default: `15`) and `--savgol-poly` (default: `3`)

| `--savgol-window` | Effect |
|-------------------|--------|
| `1` or `0` | Disabled |
| `11–15` | Default — light to moderate smoothing |
| `25–51` | Heavy smoothing for very noisy trajectories |

The window must be an odd integer and greater than `--savgol-poly + 1`. If the trajectory is too short for the chosen window, the filter is skipped automatically.

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
python monocular_vo_smoothed.py [video] [options]
```

`video` is optional — omitting it opens the default webcam (device 0).

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `video` | webcam | Path to input video file |
| `--scale` | `1.0` | Resize factor applied to each frame |
| `--features` | `500` | Max corners for Shi-Tomasi detection |
| `--min-features` | `200` | Threshold below which features are re-detected |
| `--no-fb` | off | Disable forward-backward consistency check in KLT tracking |
| `--skip` | `1` | Process every Nth frame (1 = no skip) |
| `--plot-interval` | `10` | Refresh the live trajectory plot every N frames |
| `--2d` | off | Use a 2D (X-Z) plot instead of 3D |
| `--no-temporal` | off | Skip the temporal analysis plot at the end |
| `--ema-alpha` | `0.7` | EMA smoothing factor for online translation filtering |
| `--savgol-window` | `15` | Savitzky-Golay window length (odd integer; 0 or 1 = off) |
| `--savgol-poly` | `3` | Savitzky-Golay polynomial order |

### Examples

```bash
# Run on a video file with default smoothing
python monocular_vo_smoothed.py my_drive.mp4

# Heavier smoothing for a shaky handheld video
python monocular_vo_smoothed.py handheld.mp4 --ema-alpha 0.4 --savgol-window 31

# Disable all smoothing (reproduces original behaviour)
python monocular_vo_smoothed.py my_drive.mp4 --ema-alpha 1.0 --savgol-window 1

# Downscale for speed, skip every other frame, 2D view
python monocular_vo_smoothed.py long_video.mp4 --scale 0.5 --skip 2 --2d

# Webcam with moderate smoothing
python monocular_vo_smoothed.py --ema-alpha 0.6 --savgol-window 21
```

---

## Outputs

| File | Description |
|------|-------------|
| `trajectory_research_vo.txt` | Raw XYZ trajectory, one point per line |
| `trajectory_plot.png` | Final trajectory (raw faint + smoothed bold), 2D or 3D |
| `trajectory_temporal.png` | 6-panel temporal analysis (position, velocity, speed, distance, height) |

The temporal plots use the smoothed trajectory by default. Raw values are overlaid faintly on the position and height panels.

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
  ├─ EMA smoothing on translation increment  ◄── NEW
  │
  ├─ Accumulate pose: t += smoothed_increment, R = R_rel @ R
  │
  └─ Store trajectory point

End of video
  │
  ├─ Savitzky-Golay smoothing on full trajectory  ◄── NEW
  │
  └─ Save plots + stats
```

---

## Tuning Tips

- **Video is shaky or handheld:** lower `--ema-alpha` (try `0.4–0.5`) and increase `--savgol-window` (try `25–35`).
- **Fast, smooth camera motion:** keep `--ema-alpha` near `0.7–0.8` and `--savgol-window` at `11–15` to avoid over-smoothing sharp turns.
- **Very short clips (< 30 frames):** the Savitzky-Golay filter may be skipped automatically if the window is larger than the trajectory length. Reduce `--savgol-window` accordingly.
- **Real-time performance matters:** EMA has zero overhead. If the Savitzky-Golay smoothing on the live preview is slow, increase `--plot-interval` to reduce how often the plot refreshes.
- **Comparing raw vs smoothed:** both curves are always shown in the final `trajectory_plot.png` — raw in faint blue, smoothed in solid blue.

---

## Limitations

These smoothing techniques reduce noise but do not fix the fundamental limitations of monocular VO:

- **Scale ambiguity** — there is no metric ground truth. All distances are relative.
- **Drift** — errors accumulate over time with no loop closure.
- **Degenerate motion** — pure rotation or insufficient parallax will cause pose estimation to fail regardless of smoothing.