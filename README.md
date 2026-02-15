# ZED-Based Visual SLAM and Localization Pipeline (ROS2)

## Overview

This project implements a ZED stereo camera–driven perception and localization pipeline for an autonomous rover. The system leverages the ZED SDK and ROS2 to perform stereo depth estimation, visual odometry, pose tracking, and map generation. ZED camera data serves as the primary exteroceptive sensor, integrated with IMU, GPS, and wheel encoder measurements for robust state estimation.

The pipeline is designed for real-time operation on a mobile rover platform and supports modular integration with SLAM, sensor fusion, and navigation stacks.

## Objectives

The system aims to achieve the following:

- Acquire synchronized stereo images and depth data from the ZED camera.
- Perform visual odometry using stereo vision and inertial measurements.
- Generate 3D maps and trajectories using SLAM.
- Publish pose, odometry, depth, and point clouds via ROS2 topics.
- Provide ZED-derived pose estimates for downstream sensor fusion and navigation.

This README focuses primarily on the ZED camera data pipeline.

## System Architecture

### High-Level Pipeline

```
ZED Camera
  ↓
ZED SDK
  ↓
ROS2 ZED Wrapper Nodes
  ├── Stereo Images
  ├── Depth Map
  ├── Point Cloud
  ├── IMU Data
  ├── Visual Odometry / Pose
  ↓
SLAM / Mapping / Localization Modules
```


## ZED Camera Data Acquisition

### ZED SDK Integration

The ZED SDK provides low-level access to:
- stereo image streams
- depth estimation
- inertial measurements
- visual tracking
- spatial mapping

The ZED ROS2 wrapper exposes these outputs as ROS2 topics and TF transforms.