# Visual Odometry Quick Reference

## 📦 Installation

### Prerequisites
Python 3.7+ is required. The following packages are needed:

```bash
# Install required packages
pip install opencv-python numpy matplotlib

# Or using conda
conda install -c conda-forge opencv numpy matplotlib

# For development/virtual environment
python -m venv vo_env
source vo_env/bin/activate  # On Windows: vo_env\Scripts\activate
pip install opencv-python numpy matplotlib
```

### Package Versions (Tested)
- `opencv-python >= 4.5.0`
- `numpy >= 1.19.0`
- `matplotlib >= 3.3.0`

### Verify Installation
```bash
python -c "import cv2, numpy, matplotlib; print('All packages installed successfully!')"
```

### Quick Test
```bash
# Download the script
# Run with webcam (if available)
python monocular_vo.py

# Or test with a sample video
python monocular_vo.py sample_video.mp4
```

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
--no-temporal         Skip temporal analysis plots
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
    view_3d=True,  # Enable 3D visualization
    save_temporal_plots=True  # Enable temporal analysis
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
| `trajectory_temporal.png` | Image | 6-panel temporal analysis |

### Output Details

**trajectory_research_vo.txt**: Plain text file with one row per frame
```
x1  y1  z1
x2  y2  z2
x3  y3  z3
...
```

**trajectory_plot.png**: Spatial visualization
- 3D mode: Interactive view of full X-Y-Z path
- 2D mode: Top-down X-Z projection

**trajectory_temporal.png**: Six subplots showing:
1. **Position Components**: X, Y, Z over time
2. **Velocity Components**: Vx, Vy, Vz over time
3. **Speed**: Magnitude of velocity over time
4. **Cumulative Distance**: Total path length traveled
5. **Distance from Origin**: Straight-line distance from start
6. **Height Profile**: Y-axis (vertical) position over time

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

## 📊 Interpreting Results

### Understanding the Outputs

#### 1. **3D Trajectory Plot** (`trajectory_plot.png`)
Shows the complete camera path in 3D space.

**What to look for:**
- **Smooth curves**: Indicate stable tracking
- **Jagged/erratic paths**: Suggest tracking issues or fast motion
- **Drift**: Gradual deviation from expected path (normal for monocular VO)
- **Loop closure error**: Start and end points don't match in circular paths

**Color coding:**
- 🟢 Green dot: Starting position (origin)
- 🔴 Red dot: Final position
- 🔵 Blue line: Camera trajectory

#### 2. **Temporal Analysis** (`trajectory_temporal.png`)

**Panel 1: Position Components Over Time**
- Shows X (red), Y (green), Z (blue) positions
- **Interpretation:**
  - Flat lines = no movement in that direction
  - Linear trends = constant velocity
  - Curved trends = acceleration/deceleration
  - Y drift = vertical position changes (common artifact)

**Panel 2: Velocity Components Over Time**
- Frame-to-frame displacement in each axis
- **Interpretation:**
  - Values near zero = stationary camera
  - Spikes = sudden movements
  - Oscillations = vibration or shaky footage
  - Consistent values = steady motion

**Panel 3: Speed Over Time**
- Overall movement speed (magnitude)
- **Interpretation:**
  - Peaks = fast camera motion
  - Valleys = slow/stopped camera
  - Area under curve = total distance
  - Smooth = steady recording, spiky = variable speed

**Panel 4: Cumulative Distance**
- Total path length traveled
- **Interpretation:**
  - Steep slope = fast movement period
  - Flat regions = stationary camera
  - Final value = total journey length
  - Always increasing (can't go negative)

**Panel 5: Distance from Origin**
- Straight-line distance from start
- **Interpretation:**
  - Increasing = moving away from start
  - Decreasing = returning toward start
  - Returns to zero = perfect loop closure (rare)
  - Final value = net displacement

**Panel 6: Height Profile (Y-axis)**
- Vertical position over time
- **Interpretation:**
  - Positive = camera above start point
  - Negative = camera below start point
  - Drift from zero = common in monocular VO
  - Large swings = vertical camera movement

### Terminal Output Statistics

When processing completes, you'll see:

```
==================================================
TRAJECTORY STATISTICS
==================================================
Total frames processed: 450
Total distance traveled: 125.34 m
Final distance from origin: 23.45 m
Average speed: 0.2785 m/frame
Max speed: 1.2341 m/frame
Position range - X: [-5.67, 12.34]
Position range - Y: [-2.11, 3.89]
Position range - Z: [-1.23, 45.67]
Net displacement - X: 8.21 m
Net displacement - Y: -1.34 m
Net displacement - Z: 22.10 m
==================================================
```

**What these mean:**
- **Total frames processed**: Number of trajectory points
- **Total distance traveled**: Sum of all movements (path length)
- **Final distance from origin**: Euclidean distance from start to end
- **Average/Max speed**: Movement speed in arbitrary units per frame
- **Position range**: Bounding box of trajectory
- **Net displacement**: Final position relative to start

### Common Patterns and What They Mean

#### ✅ Good Results
- Smooth trajectory with gradual curves
- Speed profile matches expected camera motion
- Low Y-axis drift (< 10% of XZ movement)
- Distance from origin matches video content
- Velocity spikes align with known fast movements

#### ⚠️ Warning Signs
- **Excessive drift**: Position keeps increasing in one direction
  - *Cause*: Insufficient features, poor lighting, scale ambiguity
  - *Solution*: Increase feature count, improve lighting, enable FB check

- **Jumpy trajectory**: Sudden position changes
  - *Cause*: Feature tracking failures, motion blur
  - *Solution*: Reduce frame skip, increase features, stabilize video

- **Y-axis domination**: Y movement >> X or Z
  - *Cause*: Common monocular VO artifact
  - *Solution*: Expected behavior, use for relative analysis only

- **Speed spikes**: Unrealistic velocity peaks
  - *Cause*: Tracking errors, reflections, moving objects
  - *Solution*: Improve scene conditions, mask dynamic objects

#### ❌ Poor Results
- Random walk pattern (no structure)
- Extreme Y drift (> 50% of total movement)
- Speed consistently > 2.0 m/frame
- Trajectory doesn't match video at all

**Causes**: Textureless scene, extreme motion blur, inadequate features, wrong camera matrix

### Scale Interpretation

⚠️ **Critical**: Monocular VO produces **relative scale only**
- Units are arbitrary (called "m" but not real meters)
- Scale drifts over time
- Compare relative movements, not absolute distances
- Cannot measure true distances without calibration

**Example**: If trajectory shows "10m", it means the camera moved 10× some baseline distance, not 10 actual meters.

### Validation Techniques

#### 1. **Visual Inspection**
```bash
# Play video and watch trajectory update
python monocular_vo.py video.mp4 --plot-interval 5
```
Match trajectory shape to camera movements visually.

#### 2. **Known Path Testing**
Record video of a known shape (square, circle) and check if trajectory matches.

#### 3. **Temporal Consistency**
```python
import numpy as np
trajectory = np.loadtxt('trajectory_research_vo.txt')

# Check for outliers
velocities = np.diff(trajectory, axis=0)
speeds = np.linalg.norm(velocities, axis=1)
outliers = speeds > (np.mean(speeds) + 3*np.std(speeds))
print(f"Outlier frames: {np.where(outliers)[0]}")
```

#### 4. **Drift Analysis**
```python
# Expected: start = (0,0,0), check final position
final_pos = trajectory[-1]
drift = np.linalg.norm(final_pos)
print(f"Total drift: {drift:.2f}")

# For loop: should return to origin
# Large drift indicates accumulation error
```

### When to Trust the Results

✅ **Trust when:**
- Trajectory shape matches visual observation
- Speed profile is reasonable (< 1.0 m/frame typical)
- Statistics are consistent across runs
- Feature count stays > 100 throughout
- No sudden trajectory jumps

⚠️ **Be cautious when:**
- Processing low-texture scenes
- High motion blur present
- Feature count drops below 50
- Large Y-axis drift observed
- Scale seems to change over time

❌ **Don't trust when:**
- Random walk behavior
- Feature count consistently < 20
- Trajectory doesn't match video at all
- Extreme jumps in position (> 10× average)

### Example Analysis Workflow

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
traj = np.loadtxt('trajectory_research_vo.txt')

# 1. Check overall quality
vel = np.diff(traj, axis=0)
speed = np.linalg.norm(vel, axis=1)
print(f"Mean speed: {speed.mean():.4f}")
print(f"Speed std: {speed.std():.4f}")
print(f"Max speed: {speed.max():.4f}")

# 2. Identify problem frames
outliers = np.where(speed > speed.mean() + 3*speed.std())[0]
print(f"Potential issues at frames: {outliers}")

# 3. Check drift
drift_per_frame = np.linalg.norm(traj, axis=1)
print(f"Total drift: {drift_per_frame[-1]:.2f}")
print(f"Drift rate: {drift_per_frame[-1]/len(traj):.4f} per frame")

# 4. Analyze direction
x_range = traj[:, 0].max() - traj[:, 0].min()
y_range = traj[:, 1].max() - traj[:, 1].min()
z_range = traj[:, 2].max() - traj[:, 2].min()
print(f"Movement ratios - X: {x_range:.2f}, Y: {y_range:.2f}, Z: {z_range:.2f}")
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
- 📉 **NEW: Temporal analysis plots** showing position, velocity, speed over time
- 📊 **NEW: Comprehensive statistics** printed to terminal
- 🔍 **NEW: Six analytical plots** for trajectory validation

## 💡 Tips for Best 3D Results

1. **Use all three axes**: The 3D view helps identify vertical drift
2. **Check plot interval**: Lower values (5-10) give smoother real-time updates
3. **Adjust viewing angle**: Rotate the 3D plot to inspect from different perspectives
4. **Compare projections**: Use the 2D projection code to analyze individual planes
5. **Monitor Y-axis**: Vertical drift is common in monocular VO
6. **Use adequate features**: More features (500-1000) improve 3D accuracy
7. **Analyze temporal plots**: Check `trajectory_temporal.png` for motion patterns
8. **Validate with statistics**: Review terminal output for quality metrics
9. **Compare speed profiles**: Ensure speed plot matches expected camera motion
10. **Check cumulative distance**: Should align with visual path length estimate