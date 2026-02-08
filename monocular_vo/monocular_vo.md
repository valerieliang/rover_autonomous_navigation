# Visual Odometry Quick Reference

## 🚀 Quick Start

```bash
# Webcam (3D visualization)
python monocular_vo.py

# Video file (3D visualization)
python monocular_vo.py video.mp4

# 2D visualization (legacy mode)
python monocular_vo.py video.mp4 --2d

# Fast mode (3D)
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
--2d                  Use 2D visualization (X-Z plane only)
```

## 📊 Performance Presets

### Real-Time 3D (50-80 fps)
```bash
python monocular_vo.py video.mp4 \
    --scale 0.5 --no-fb --skip 2 --features 300
```

### Balanced 3D (20-35 fps)
```bash
python monocular_vo.py video.mp4 \
    --scale 0.75 --features 500
```

### High Quality 3D (8-15 fps)
```bash
python monocular_vo.py video.mp4 \
    --scale 1.0 --features 1000 --min-features 400
```

### Legacy 2D Mode (faster)
```bash
python monocular_vo.py video.mp4 --2d \
    --scale 0.75 --features 500
```

## 🐍 Python API

### Basic Usage (3D)
```python
from monocular_vo import monocular_vo

# Default 3D visualization
monocular_vo(video_path="input.mp4")

# 2D visualization
monocular_vo(video_path="input.mp4", view_3d=False)
```

### Custom Parameters
```python
monocular_vo(
    video_path="input.mp4",
    scale_factor=0.75,
    max_features=500,
    use_fb_check=True,
    frame_skip=1,
    plot_interval=10,
    view_3d=True  # Enable 3D visualization
)
```

### Individual Functions
```python
from monocular_vo import (
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
| `trajectory_plot.png` | Image | Final trajectory plot (2D or 3D) |

### Loading Trajectory
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load trajectory
trajectory = np.loadtxt('trajectory_research_vo.txt')
x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# Compute statistics
total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
print(f"Total distance: {total_distance:.2f}")

# 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, '-b', linewidth=2)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.show()

# 2D projections
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(x, y, '-b'); axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
axes[1].plot(x, z, '-b'); axes[1].set_xlabel('X'); axes[1].set_ylabel('Z')
axes[2].plot(y, z, '-b'); axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z')
plt.tight_layout()
plt.show()
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

### Visualization Mode
- **3D (default)**: Full trajectory in X-Y-Z space
- **2D (`--2d`)**: X-Z plane only, slightly faster

## ⚡ Performance Matrix

| Resolution | Default 3D | Fast 3D | Fastest 3D | 2D Mode |
|------------|-----------|---------|-----------|---------|
| 1080p | 12 fps | 35 fps | 75 fps | 15 fps |
| 720p | 25 fps | 65 fps | 130 fps | 30 fps |
| 480p | 50 fps | 110 fps | 190 fps | 60 fps |

**Default:** Standard settings with 3D visualization  
**Fast:** `--scale 0.75 --no-fb`  
**Fastest:** `--scale 0.5 --no-fb --skip 2 --features 300`  
**2D Mode:** Add `--2d` flag for traditional X-Z visualization

## 🔧 Troubleshooting

### Too Few Features
```bash
# Reduce threshold or increase max
python monocular_vo.py video.mp4 \
    --min-features 150 --features 800
```

### Erratic Trajectory
```bash
# Enable FB check, reduce skip
python monocular_vo.py video.mp4 \
    --skip 1  # Remove --no-fb if present
```

### Too Slow (3D rendering)
```bash
# Use 2D mode or reduce resolution
python monocular_vo.py video.mp4 --2d \
    --scale 0.5 --features 300 --no-fb

# Or increase plot interval
python monocular_vo.py video.mp4 \
    --plot-interval 20 --scale 0.75
```

### Camera Access Issues
```python
# Try different camera index
cap = cv2.VideoCapture(1)  # Instead of 0
```

### 3D Plot Not Showing
```bash
# Ensure matplotlib has 3D support
pip install matplotlib --upgrade

# Or use 2D mode
python monocular_vo.py video.mp4 --2d
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
- **3D Mode**: Full X-Y-Z trajectory with interactive rotation
- **2D Mode**: Top-down X-Z view (legacy)

## 🎨 3D Visualization Features

### Interactive Controls
- **Rotate**: Click and drag
- **Zoom**: Mouse wheel or scroll
- **Pan**: Right-click and drag (or Ctrl+click on Mac)

### Viewing Angles
The default view is set to:
- Elevation: 20° (looking slightly down)
- Azimuth: 45° (diagonal view)

### Visual Elements
- 🔵 **Blue line**: Camera trajectory path
- 🟢 **Green marker**: Starting position
- 🔴 **Red marker**: Current/ending position
- Grid lines for spatial reference
- Equal aspect ratio (prevents distortion)

## 🔬 Algorithm Summary

```
Video → Grayscale → Feature Detection → Tracking (LK Flow)
                                             ↓
                                    Pose Estimation (Essential Matrix)
                                             ↓
                                    Scale Estimation (heuristic)
                                             ↓
                                    3D Trajectory Integration
                                             ↓
                                    3D Visualization & Output
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

**3D Pose Recovery:** Decomposes Essential Matrix
```
E = [t]ₓ R
where R ∈ SO(3), t ∈ ℝ³
```

## ⚠️ Limitations

- ❌ No absolute scale (monocular)
- ❌ Drift accumulates over time
- ❌ Requires textured scenes
- ❌ Sensitive to motion blur
- ❌ No loop closure
- ⚠️ Y-axis drift more pronounced than X/Z
- ⚠️ 3D visualization requires more GPU/CPU

## ✅ Best Practices

1. Use 3D visualization for full spatial understanding
2. Enable forward-backward check for robustness
3. Process all frames for best accuracy
4. Increase features for longer sequences
5. Use 2D mode for faster performance on slower hardware
6. Calibrate camera if possible
7. Consider stereo/IMU for metric scale

## 📞 Common Use Cases

### Robotics Navigation (3D)
```bash
python monocular_vo.py /dev/video0 \
    --scale 0.5 --features 500
```

### Drone/Aerial Analysis (3D recommended)
```bash
python monocular_vo.py flight.mp4 \
    --scale 1.0 --features 1000
```

### Video Analysis (High Quality 3D)
```bash
python monocular_vo.py dataset.mp4 \
    --scale 1.0 --features 1000
```

### Real-Time Demo (Fast 3D)
```bash
python monocular_vo.py \
    --scale 0.5 --no-fb --features 300
```

### Legacy 2D Analysis
```bash
python monocular_vo.py video.mp4 --2d \
    --scale 0.75 --features 500
```

## 🆕 What's New in 3D Version

- ✨ Full 3D trajectory visualization with X, Y, Z components
- 🎯 Interactive 3D plots with rotation/zoom controls
- 📊 Better spatial understanding of camera motion
- 🎨 Enhanced visual markers for start/end positions
- ⚡ Equal aspect ratio for accurate spatial representation
- 🔄 Backward compatible with 2D mode via `--2d` flag
- 📈 Improved final plot with larger, clearer markers

## 💡 Tips for Best 3D Results

1. **Use all three axes**: The 3D view helps identify vertical drift
2. **Check plot interval**: Lower values (5-10) give smoother real-time updates
3. **Adjust viewing angle**: Rotate the 3D plot to inspect from different perspectives
4. **Compare projections**: Use the 2D projection code to analyze individual planes
5. **Monitor Y-axis**: Vertical drift is common in monocular VO
6. **Use adequate features**: More features (500-1000) improve 3D accuracy