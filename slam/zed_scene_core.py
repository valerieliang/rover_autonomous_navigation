"""
zed_scene_understanding.py  —  Wall/Hallway Detection + Line-of-Sight Object Validation
========================================================================================
Uses the ZED depth map + left RGB image to:

  1. WALL DETECTION      — segments the depth frame into planar surfaces,
                           classifies left / right / front walls
  2. HALLWAY DETECTION   — infers a hallway when left + right walls are
                           roughly parallel and the centre is open
  3. LINE-OF-SIGHT PROBE — samples depth along the camera's forward axis
                           and reports what (if anything) is directly ahead
  4. OBJECT VALIDATION   — draws bounding boxes around depth clusters in
                           the forward corridor so you can place a physical
                           object and confirm the system sees it

All output is overlaid on the live camera feed and printed to the console.

Usage
-----
  python zed_scene_understanding.py               # live camera
  python zed_scene_understanding.py --svo f.svo   # SVO replay
  python zed_scene_understanding.py --no-display  # headless
  python zed_scene_understanding.py --help
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("[ERROR] pyzed not found. Install the ZED SDK.")


# =============================================================================
# Tuneable parameters  (edit these to match your environment)
# =============================================================================

class Config:
    # ── Depth filtering ───────────────────────────────────────────────────────
    DEPTH_MIN_M        = 0.3    # ignore anything closer than this [m]
    DEPTH_MAX_M        = 8.0    # ignore anything farther than this [m]

    # ── Wall detection ────────────────────────────────────────────────────────
    # The depth image is divided into a grid of cells; each cell votes on
    # whether it looks like a flat surface (low depth variance = wall-like).
    GRID_COLS          = 12     # horizontal cells
    GRID_ROWS          = 8      # vertical cells
    WALL_VAR_THRESH    = 0.04   # max depth variance (m²) for a "flat" cell
    WALL_MEAN_MAX_M    = 5.0    # max mean depth for a cell to be called a wall

    # ── Hallway detection ─────────────────────────────────────────────────────
    # A hallway is detected when:
    #   - left-third AND right-third of the image have close flat surfaces
    #   - centre-third mean depth is significantly larger (open corridor)
    HALLWAY_SIDE_MAX_M = 2.5    # side walls must be closer than this
    HALLWAY_CENTRE_MIN_M = 1.2  # centre must be this much farther than sides
    HALLWAY_OPEN_RATIO = 1.5    # centre_depth / side_depth > this → hallway

    # ── Line-of-sight probe ───────────────────────────────────────────────────
    # Samples a central rectangle (fraction of frame size)
    LOS_WIDTH_FRAC     = 0.15   # half-width of probe region (fraction of W)
    LOS_HEIGHT_FRAC    = 0.15   # half-height of probe region
    LOS_OBJECT_MAX_M   = 4.0    # report object if median depth < this

    # ── Object cluster detection (in forward corridor) ────────────────────────
    CLUSTER_DEPTH_MAX_M = 4.0   # only look for objects within this range
    CLUSTER_MIN_PIXELS  = 400   # minimum cluster size to report
    CLUSTER_DILATE_ITER = 3     # morphological dilation before clustering

    # ── Display ───────────────────────────────────────────────────────────────
    OVERLAY_ALPHA      = 0.45


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclass
class WallInfo:
    left_wall:    bool  = False
    right_wall:   bool  = False
    front_wall:   bool  = False
    left_dist_m:  float = float("inf")
    right_dist_m: float = float("inf")
    front_dist_m: float = float("inf")


@dataclass
class HallwayInfo:
    detected:      bool  = False
    width_est_m:   float = 0.0   # rough corridor width estimate
    centre_open_m: float = 0.0   # depth of open centre


@dataclass
class LOSObject:
    detected:  bool  = False
    dist_m:    float = float("inf")
    label:     str   = ""        # future: plug in a classifier here


@dataclass
class SceneState:
    walls:   WallInfo   = field(default_factory=WallInfo)
    hallway: HallwayInfo = field(default_factory=HallwayInfo)
    los_obj: LOSObject  = field(default_factory=LOSObject)
    frame_idx: int      = 0


# =============================================================================
# Core analysis functions
# =============================================================================

def preprocess_depth(depth_np: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Replace invalid / out-of-range depth values with NaN.
    Input:  raw float32 depth array from sl.Mat.get_data()
    Output: cleaned float32 array, same shape
    """
    d = depth_np.copy().astype(np.float32)
    d[~np.isfinite(d)]          = np.nan
    d[d < cfg.DEPTH_MIN_M]      = np.nan
    d[d > cfg.DEPTH_MAX_M]      = np.nan
    return d


def nan_mean(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    return float(np.mean(v)) if len(v) > 0 else float("inf")


def nan_var(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    return float(np.var(v)) if len(v) > 1 else float("inf")


def detect_walls(depth: np.ndarray, cfg: Config) -> WallInfo:
    """
    Divide the depth image into a grid, mark flat cells, then
    aggregate left / centre / right columns to classify walls.
    """
    h, w = depth.shape
    ch   = h // cfg.GRID_ROWS
    cw   = w // cfg.GRID_COLS

    # Build flatness map: True where cell looks like a wall
    flat = np.zeros((cfg.GRID_ROWS, cfg.GRID_COLS), dtype=bool)
    mean = np.full((cfg.GRID_ROWS, cfg.GRID_COLS), float("inf"))

    for r in range(cfg.GRID_ROWS):
        for c in range(cfg.GRID_COLS):
            cell = depth[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            m    = nan_mean(cell)
            v    = nan_var(cell)
            mean[r, c] = m
            if v < cfg.WALL_VAR_THRESH and m < cfg.WALL_MEAN_MAX_M:
                flat[r, c] = True

    # Column thirds
    third = cfg.GRID_COLS // 3
    left_cols   = slice(0, third)
    centre_cols = slice(third, 2 * third)
    right_cols  = slice(2 * third, cfg.GRID_COLS)

    # A "zone" has a wall if >40% of its cells are flat
    def zone_wall(col_slice):
        region = flat[:, col_slice]
        frac   = region.sum() / region.size
        dist   = nan_mean(mean[:, col_slice])
        return frac > 0.40, dist

    left_wall,   left_d   = zone_wall(left_cols)
    front_wall,  front_d  = zone_wall(centre_cols)
    right_wall,  right_d  = zone_wall(right_cols)

    return WallInfo(
        left_wall    = left_wall,
        right_wall   = right_wall,
        front_wall   = front_wall,
        left_dist_m  = left_d,
        right_dist_m = right_d,
        front_dist_m = front_d,
    )


def detect_hallway(depth: np.ndarray, walls: WallInfo, cfg: Config) -> HallwayInfo:
    """
    Infer a hallway from left + right walls being close and parallel,
    with an open centre.
    """
    h, w  = depth.shape
    third = w // 3

    left_strip   = depth[:, :third]
    centre_strip = depth[:, third:2*third]
    right_strip  = depth[:, 2*third:]

    left_d   = nan_mean(left_strip)
    centre_d = nan_mean(centre_strip)
    right_d  = nan_mean(right_strip)

    side_d = min(left_d, right_d)

    hallway = (
        walls.left_wall
        and walls.right_wall
        and left_d  < cfg.HALLWAY_SIDE_MAX_M
        and right_d < cfg.HALLWAY_SIDE_MAX_M
        and centre_d > side_d * cfg.HALLWAY_OPEN_RATIO
        and (centre_d - side_d) > cfg.HALLWAY_CENTRE_MIN_M
    )

    # Rough width: if we know the camera HFOV (~90° for ZED 2 at HD720)
    # width ≈ 2 * side_dist * tan(FOV/6)  (for one third of frame)
    hfov_rad   = math.radians(90)
    width_est  = 2 * side_d * math.tan(hfov_rad / 6) if side_d < float("inf") else 0.0

    return HallwayInfo(
        detected      = hallway,
        width_est_m   = width_est,
        centre_open_m = centre_d,
    )


def probe_line_of_sight(depth: np.ndarray, cfg: Config) -> LOSObject:
    """
    Sample depth in a central rectangle.
    If something is there closer than LOS_OBJECT_MAX_M, flag it.
    """
    h, w = depth.shape
    cx, cy = w // 2, h // 2
    dw = int(w * cfg.LOS_WIDTH_FRAC)
    dh = int(h * cfg.LOS_HEIGHT_FRAC)

    roi = depth[cy-dh:cy+dh, cx-dw:cx+dw]
    valid = roi[np.isfinite(roi)]

    if len(valid) == 0:
        return LOSObject(detected=False)

    med = float(np.median(valid))

    if med < cfg.LOS_OBJECT_MAX_M:
        return LOSObject(
            detected = True,
            dist_m   = med,
            label    = f"Object @ {med:.2f} m",
        )
    return LOSObject(detected=False, dist_m=med)


def find_forward_clusters(depth: np.ndarray, image_bgr: np.ndarray,
                          cfg: Config) -> list[tuple]:
    """
    Find depth clusters in the forward corridor (centre half of the frame)
    that are within CLUSTER_DEPTH_MAX_M.
    Returns list of (x, y, w, h, mean_depth) bounding boxes.
    """
    h, w = depth.shape

    # Mask: valid depth within range, centre horizontal band
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth < cfg.CLUSTER_DEPTH_MAX_M)
    mask[valid] = 255

    # Focus on centre corridor (middle 60% width)
    side_crop = w // 5
    mask[:, :side_crop]  = 0
    mask[:, w-side_crop:] = 0

    # Morphological closing to merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.dilate(mask, kernel, iterations=cfg.CLUSTER_DILATE_ITER)
    mask   = cv2.erode(mask,  kernel, iterations=cfg.CLUSTER_DILATE_ITER)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    boxes = []
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < cfg.CLUSTER_MIN_PIXELS:
            continue
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        cluster_depth = depth[labels == i]
        mean_d = nan_mean(cluster_depth)
        boxes.append((x, y, bw, bh, mean_d))

    # Sort by distance (closest first)
    boxes.sort(key=lambda b: b[4])
    return boxes


# =============================================================================
# Visualisation
# =============================================================================

def colorise_depth(depth: np.ndarray, max_m: float = 6.0) -> np.ndarray:
    """Convert float32 depth → BGR colour image for display."""
    norm = np.clip(depth / max_m, 0, 1)
    norm[~np.isfinite(norm)] = 0
    grey = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)


def draw_overlay(frame: np.ndarray, depth: np.ndarray,
                 scene: SceneState, clusters: list, cfg: Config) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Semi-transparent side wall highlights ─────────────────────────────────
    overlay = out.copy()
    third   = w // 3

    if scene.walls.left_wall:
        cv2.rectangle(overlay, (0, 0), (third, h), (255, 100, 0), -1)
    if scene.walls.right_wall:
        cv2.rectangle(overlay, (2*third, 0), (w, h), (255, 100, 0), -1)
    if scene.walls.front_wall:
        cv2.rectangle(overlay, (third, 0), (2*third, h), (0, 60, 255), -1)

    cv2.addWeighted(overlay, cfg.OVERLAY_ALPHA * 0.4, out, 1 - cfg.OVERLAY_ALPHA * 0.4, 0, out)

    # ── LOS probe rectangle ───────────────────────────────────────────────────
    cx, cy = w // 2, h // 2
    dw = int(w * cfg.LOS_WIDTH_FRAC)
    dh = int(h * cfg.LOS_HEIGHT_FRAC)
    los_color = (0, 255, 0) if scene.los_obj.detected else (180, 180, 180)
    cv2.rectangle(out, (cx-dw, cy-dh), (cx+dw, cy+dh), los_color, 2)

    # ── Object cluster boxes ──────────────────────────────────────────────────
    for idx, (x, y, bw, bh, mean_d) in enumerate(clusters):
        cv2.rectangle(out, (x, y), (x+bw, y+bh), (0, 255, 200), 2)
        cv2.putText(out, f"{mean_d:.2f}m", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

    # ── HUD panel ─────────────────────────────────────────────────────────────
    panel_h = 200
    panel   = np.zeros((panel_h, w, 3), dtype=np.uint8)

    def put(text, row, color=(200, 220, 200)):
        cv2.putText(panel, text, (10, 22 + row * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

    # Wall status
    walls = scene.walls
    lc = (0, 255, 120) if walls.left_wall  else (80, 80, 80)
    rc = (0, 255, 120) if walls.right_wall else (80, 80, 80)
    fc = (0, 100, 255) if walls.front_wall else (80, 80, 80)

    put(f"LEFT wall : {'YES' if walls.left_wall  else 'no'}  {walls.left_dist_m:.2f} m",  0, lc)
    put(f"RIGHT wall: {'YES' if walls.right_wall else 'no'}  {walls.right_dist_m:.2f} m", 1, rc)
    put(f"FRONT wall: {'YES' if walls.front_wall else 'no'}  {walls.front_dist_m:.2f} m", 2, fc)

    # Hallway
    hw = scene.hallway
    hc = (0, 255, 255) if hw.detected else (80, 80, 80)
    hw_txt = (f"HALLWAY  width~{hw.width_est_m:.1f}m  "
              f"open {hw.centre_open_m:.1f}m ahead"
              if hw.detected else "Hallway: not detected")
    put(hw_txt, 3, hc)

    # LOS object
    lo = scene.los_obj
    lc2 = (0, 255, 0) if lo.detected else (80, 80, 80)
    lo_txt = (f"LOS OBJECT: {lo.label}" if lo.detected
              else f"LOS: clear  ({lo.dist_m:.2f} m)")
    put(lo_txt, 4, lc2)

    put(f"Clusters in corridor: {len(clusters)}   Frame: {scene.frame_idx}", 5)

    out = np.vstack([out, panel])
    return out


# =============================================================================
# Aliases for zed_slam_main.py compatibility
# =============================================================================

#: ``SceneConfig`` is the public name expected by ``zed_slam_main.py``.
#: ``Config`` is kept for backwards compatibility with the standalone script.
SceneConfig = Config

def draw_scene_overlay(frame: np.ndarray, depth: np.ndarray,
                       scene: SceneState, clusters: list,
                       cfg: Config) -> np.ndarray:
    """Thin wrapper around :func:`draw_overlay` — name used by zed_slam_main."""
    return draw_overlay(frame, depth, scene, clusters, cfg)


# =============================================================================
# Main loop
# =============================================================================

def run(args: argparse.Namespace):
    cfg = Config()

    # ── ZED init ──────────────────────────────────────────────────────────────
    zed = sl.Camera()

    p = sl.InitParameters()
    p.camera_resolution      = sl.RESOLUTION.HD720
    p.camera_fps             = 30
    p.depth_mode             = sl.DEPTH_MODE.ULTRA
    p.coordinate_units       = sl.UNIT.METER
    p.depth_minimum_distance = cfg.DEPTH_MIN_M
    p.depth_maximum_distance = cfg.DEPTH_MAX_M

    if args.svo:
        p.set_from_svo_file(str(args.svo))
        p.svo_real_time_mode = False

    err = zed.open(p)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"[ERROR] Cannot open ZED: {err}")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50

    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    scene    = SceneState()
    frame_idx = 0

    print("\nZED Scene Understanding running.")
    print("Press Q or ESC to quit.\n")
    print(f"{'Frame':>6}  {'L-wall':>8}  {'R-wall':>8}  {'F-wall':>8}  "
          f"{'Hallway':>8}  {'LOS dist':>9}  {'Clusters':>8}")
    print("-" * 70)

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            if args.svo:
                print("[INFO] SVO playback complete.")
                break
            continue

        # ── Retrieve data ─────────────────────────────────────────────────────
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

        frame_bgr = image_zed.get_data()[:, :, :3].copy()
        depth_raw = depth_zed.get_data().copy()

        depth = preprocess_depth(depth_raw, cfg)

        # ── Analysis ──────────────────────────────────────────────────────────
        walls    = detect_walls(depth, cfg)
        hallway  = detect_hallway(depth, walls, cfg)
        los_obj  = probe_line_of_sight(depth, cfg)
        clusters = find_forward_clusters(depth, frame_bgr, cfg)

        scene = SceneState(
            walls     = walls,
            hallway   = hallway,
            los_obj   = los_obj,
            frame_idx = frame_idx,
        )

        # ── Console ───────────────────────────────────────────────────────────
        if frame_idx % 15 == 0:
            print(
                f"{frame_idx:>6}  "
                f"{'Y '+f'{walls.left_dist_m:.1f}m' if walls.left_wall  else 'N':>8}  "
                f"{'Y '+f'{walls.right_dist_m:.1f}m' if walls.right_wall else 'N':>8}  "
                f"{'Y '+f'{walls.front_dist_m:.1f}m' if walls.front_wall else 'N':>8}  "
                f"{'YES' if hallway.detected else 'no':>8}  "
                f"{los_obj.dist_m:>9.2f}  "
                f"{len(clusters):>8}"
            )

        # ── Display ───────────────────────────────────────────────────────────
        if not args.no_display:
            vis = draw_overlay(frame_bgr, depth, scene, clusters, cfg)

            # Side-by-side depth colormap
            depth_col = colorise_depth(depth, max_m=cfg.DEPTH_MAX_M)
            # Pad depth image to match vis height
            pad = vis.shape[0] - depth_col.shape[0]
            depth_col = np.vstack([
                depth_col,
                np.zeros((pad, depth_col.shape[1], 3), dtype=np.uint8)
            ])
            combined = np.hstack([vis, depth_col])
            combined = cv2.resize(combined, None, fx=0.75, fy=0.75)

            cv2.imshow("ZED Scene Understanding", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] User quit.")
                break

        frame_idx += 1

    cv2.destroyAllWindows()
    zed.close()
    print("\nDone.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZED wall/hallway detection + line-of-sight object validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--svo",        type=Path, default=None,
                        help="SVO file for playback (omit for live camera)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — no OpenCV windows")
    args = parser.parse_args()
    run(args)