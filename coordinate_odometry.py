# Simulate the differential (4-wheel averaged) odometry and convert final local x,y into latitude/longitude.
import math
from dataclasses import dataclass

R_EARTH = 6378137.0  # mean Earth radius in meters (WGS84 approx)

@dataclass
class Pose:
    x: float    # east (meters)
    y: float    # north (meters)
    theta: float  # radians, 0 = east

class RobotOdometer:
    def __init__(self, wheel_diameter, baseline, start_lat=0.0, start_lon=0.0,
                 x=0.0, y=0.0, heading_deg=0.0):
        """
        wheel_diameter: meters
        baseline: distance between left and right wheels (meters)
        start_lat, start_lon: starting geographic coordinates in degrees
        x, y: initial local coordinates in meters (east, north)
        heading_deg: initial heading in degrees (0 = east, positive CCW)
        """
        self.wheel_diameter = wheel_diameter
        self.wheel_circumference = math.pi * wheel_diameter
        self.baseline = baseline
        self.pose = Pose(x, y, math.radians(heading_deg))
        # Geographic
        self.lat = start_lat
        self.lon = start_lon

    def _rotations_to_distances(self, rotations):
        # rotations: dict with keys 'fl','rl','fr','rr'
        s_fl = rotations['fl'] * self.wheel_circumference
        s_rl = rotations['rl'] * self.wheel_circumference
        s_fr = rotations['fr'] * self.wheel_circumference
        s_rr = rotations['rr'] * self.wheel_circumference
        return s_fl, s_rl, s_fr, s_rr

    def step(self, rotations):
        """
        Apply one odometry step given wheel rotations for all four wheels.
        Updates local pose (x,y,theta) and geographic coordinates (lat,lon).
        Returns: delta_s (m), new_pose, new_lat_lon
        """
        s_fl, s_rl, s_fr, s_rr = self._rotations_to_distances(rotations)
        s_left = (s_fl + s_rl) / 2.0
        s_right = (s_fr + s_rr) / 2.0

        # Differential-drive kinematics
        delta_theta = (s_right - s_left) / self.baseline  # radians
        delta_s = (s_left + s_right) / 2.0  # meters

        # Local displacement in robot frame -> world frame
        dx = delta_s * math.cos(self.pose.theta + delta_theta / 2.0)
        dy = delta_s * math.sin(self.pose.theta + delta_theta / 2.0)

        # Update local pose
        self.pose.x += dx
        self.pose.y += dy
        self.pose.theta = (self.pose.theta + delta_theta)  # keep radians

        # Update geographic coords using current latitude approximation
        # x is east, y is north
        # delta_lat = dy / R  (radians) -> degrees
        delta_lat_deg = (dy / R_EARTH) * (180.0 / math.pi)
        # delta_lon uses latitude scaling
        lat_rad = math.radians(self.lat)
        # Use current latitude for conversion (small-step approx)
        delta_lon_deg = (dx / (R_EARTH * math.cos(lat_rad))) * (180.0 / math.pi)

        # Update lat/lon
        self.lat += delta_lat_deg
        self.lon += delta_lon_deg

        return delta_s, (self.pose.x, self.pose.y, math.degrees(self.pose.theta) % 360), (self.lat, self.lon)

    def run_sequence(self, rotations_list):
        """
        rotations_list: list of rotations dicts to apply sequentially.
        Returns final local pose and lat/lon.
        """
        for rot in rotations_list:
            self.step(rot)
        return self.pose, (self.lat, self.lon)

    def final_coordinates_rounded(self, decimals=5):
        # decimals=5 gives ~1.1 m precision for latitude; good for "meter accuracy" rounding
        return round(self.lat, decimals), round(self.lon, decimals)

# Example usage with provided rotations
if __name__ == "__main__":
    WHEEL_DIAMETER = 0.1   # meters
    BASELINE = 0.3         # meters (left-right distance)
    START_LAT = 39.0
    START_LON = -76.6

    # Single-step example (use previous example rotations)
    ROTATIONS = {"fl": 5, "rl": 5, "fr": 6, "rr": 6}

    odo = RobotOdometer(WHEEL_DIAMETER, BASELINE, start_lat=START_LAT, start_lon=START_LON,
                       x=0.0, y=0.0, heading_deg=0.0)
    delta_s, local_pose, latlon = odo.step(ROTATIONS)

    print("=== Single step result ===")
    print(f"Delta distance (avg): {delta_s:.4f} m")
    print(f"Local pose (x east, y north, heading deg): x={local_pose[0]:.4f} m, y={local_pose[1]:.4f} m, heading={local_pose[2]:.4f}°")
    print(f"Geographic position (degrees): lat={latlon[0]:.8f}, lon={latlon[1]:.8f}")
    print(f"Rounded to meter-accuracy (~5 decimals): {odo.final_coordinates_rounded(decimals=5)}")

    # Example: run a small sequence of steps
    odo2 = RobotOdometer(WHEEL_DIAMETER, BASELINE, start_lat=START_LAT, start_lon=START_LON,
                        heading_deg=10.0)  # start heading 10° CCW from east
    seq = [
        {"fl": 3, "rl": 3, "fr": 3, "rr": 3},
        {"fl": 0, "rl": 0, "fr": 1, "rr": 1},  # turn right a bit (right wheels rotate)
        {"fl": 2, "rl": 2, "fr": 2, "rr": 2},
    ]
    pose_after, latlon_after = odo2.run_sequence(seq)
    print("\n=== Sequence result ===")
    print(f"Final local pose: x={pose_after.x:.4f} m, y={pose_after.y:.4f} m, heading={math.degrees(pose_after.theta)%360:.4f}°")
    print(f"Final geographic (deg): lat={latlon_after[0]:.8f}, lon={latlon_after[1]:.8f}")
    print(f"Rounded to meter-accuracy (~5 decimals): {odo2.final_coordinates_rounded(decimals=5)}")
