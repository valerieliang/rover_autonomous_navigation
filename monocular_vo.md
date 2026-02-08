# Visual Odometry Quick Reference

## 🚀 Quick Start

```bash
# Webcam
python monocular_vo.py

# Video file
python monocular_vo.py video.mp4

# Fast mode
python monocular_vo.py video.mp4 --scale 0.5 --no-fb --skip 2
```

## ⚙️ Command-Line Options

```
--scale FLOAT         Frame scaling (0.25-1.0, default: 1.0)
--features INT        Max features (200-2000, default: 500)
--min-features INT    Min before re-detect (default: 200)
--no-fb               Disable forward-backward check
--skip INT            Process every Nth frame (default: 1)
--plot-interval INT   Frames between plots (default: 10)
```

## 📊 Performance Presets

### Real-Time (60-100 fps)
```bash
python monocular_vo_documented.py video.mp4 \
    --scale 0.5 --no-fb --skip 2 --features 300
```

### Balanced (25-40 fps)
```bash
python monocular_vo_documented.py video.mp4 \
    --scale 0.75 --features 500
```

### High Quality (10-20 fps)
```bash
python monocular_vo_documented.py video.mp4 \
    --scale 1.0 --features 1000 --min-features 400
```

## 🐍 Python API

### Basic Usage
```python
from monocular_vo_documented import monocular_vo

monocular_vo(video_path="input.mp4")
```

### Custom Parameters
```python
monocular_vo(
    video_path="input.mp4",
    scale_factor=0.75,
    max_features=500,
    use_fb_check=True,
    frame_skip=1
)
```

### Individual Functions
```python
from monocular_vo_documented import (
    get_camera_matrix,
    track_features,
    estimate_pose,
    estimate_scale
)

# Camera calibration
K = get_camera_matrix(1920, 1080, fov_deg=70)

# Feature tracking
import cv2
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, 500, 0.01, 7)
prev_good, curr_good = track_features(prev_gray, curr_gray, prev_pts)

# Pose estimation
R, t = estimate_pose(prev_good, curr_good, K)

# Scale estimation
scale = estimate_scale(prev_good, curr_good)
```

## 📁 Output Files

| File | Format | Description |
|------|--------|-------------|
| `trajectory_research_vo.txt` | Text | Nx3 trajectory (x, y, z) |
| `trajectory_plot.png` | Image | Final trajectory plot |

### Loading Trajectory
```python
import numpy as np

# Load trajectory
trajectory = np.loadtxt('trajectory_research_vo.txt')
x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# Compute statistics
total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
print(f"Total distance: {total_distance:.2f}")
```

## 🎯 Parameter Selection Guide

### Resolution Scaling (`--scale`)
- **1.0**: Full quality, slowest
- **0.75**: Minor quality loss, 2x faster
- **0.5**: Good balance, 4x faster
- **0.25**: Lower quality, 8x faster

### Feature Count (`--features`)
- **200-300**: Fast, acceptable quality
- **500**: Balanced (recommended)
- **1000+**: High quality, slower

### Frame Skip (`--skip`)
- **1**: Process all frames (best quality)
- **2**: Skip half, 2x faster
- **3+**: Fast but may lose tracking

### Forward-Backward Check
- **Enabled (default)**: Robust, 2x slower
- **Disabled (`--no-fb`)**: Fast, less robust

## ⚡ Performance Matrix

| Resolution | Default | Fast | Fastest |
|------------|---------|------|---------|
| 1080p | 15 fps | 40 fps | 85 fps |
| 720p | 30 fps | 75 fps | 140 fps |
| 480p | 60 fps | 120 fps | 200+ fps |

**Default:** Standard settings  
**Fast:** `--scale 0.75 --no-fb`  
**Fastest:** `--scale 0.5 --no-fb --skip 2 --features 300`

## 🔧 Troubleshooting

### Too Few Features
```bash
# Reduce threshold or increase max
python monocular_vo_documented.py video.mp4 \
    --min-features 150 --features 800
```

### Erratic Trajectory
```bash
# Enable FB check, reduce skip
python monocular_vo_documented.py video.mp4 \
    --skip 1  # Remove --no-fb if present
```

### Too Slow
```bash
# Reduce resolution and features
python monocular_vo_documented.py video.mp4 \
    --scale 0.5 --features 300 --no-fb
```

### Camera Access Issues
```python
# Try different camera index
cap = cv2.VideoCapture(1)  # Instead of 0
```

## 📐 Coordinate System

```
    Y (up)
    |
    |
    |_______ X (right)
   /
  /
 Z (forward)
```

- Origin: Starting camera position
- Units: Arbitrary (no metric scale)

## 🔬 Algorithm Summary

```
Video → Grayscale → Feature Detection → Tracking (LK Flow)
                                             ↓
                                    Pose Estimation (Essential Matrix)
                                             ↓
                                    Scale Estimation (heuristic)
                                             ↓
                                    Trajectory Integration
                                             ↓
                                    Output & Visualization
```

## 📚 Key Concepts

**Essential Matrix:** Relates corresponding points between two views
```
p₂ᵀ E p₁ = 0
```

**Lucas-Kanade Flow:** Assumes brightness constancy
```
I₁(x) = I₂(x + d)
```

**RANSAC:** Robust estimation with outlier rejection
```
N = log(1-p) / log(1-wᵏ)
```

## ⚠️ Limitations

- ❌ No absolute scale (monocular)
- ❌ Drift accumulates over time
- ❌ Requires textured scenes
- ❌ Sensitive to motion blur
- ❌ No loop closure

## ✅ Best Practices

1. Use forward-backward check for robustness
2. Process all frames for best accuracy
3. Increase features for longer sequences
4. Calibrate camera if possible
5. Consider stereo/IMU for metric scale

## 📞 Common Use Cases

### Robotics Navigation
```bash
python monocular_vo_documented.py /dev/video0 \
    --scale 0.5 --features 500
```

### Video Analysis
```bash
python monocular_vo_documented.py dataset.mp4 \
    --scale 1.0 --features 1000
```

### Real-Time Demo
```bash
python monocular_vo_documented.py \
    --scale 0.5 --no-fb --features 300
```
