import math

# PARAMETERS
WHEEL_DIAMETER = 0.1   # meters
BASELINE = 0.3         # distance between left and right wheels (2L), meters

# Example rotations for each wheel (front-left, rear-left, front-right, rear-right)
ROTATIONS = {
    "fl": 5,   # front left
    "rl": 5,   # rear left
    "fr": 6,   # front right
    "rr": 6    # rear right
}

class RobotOdometer:
    def __init__(self, wheel_diameter, baseline, x=0.0, y=0.0, heading=0.0):
        """
        wheel_diameter: in meters
        baseline: distance between left and right wheels (2L)
        x, y: initial position
        heading: initial orientation in degrees (0 = facing east)
        """
        self.wheel_diameter = wheel_diameter
        self.wheel_circumference = math.pi * wheel_diameter
        self.baseline = baseline
        self.x = x
        self.y = y
        self.heading = math.radians(heading)  # store in radians

    def travel(self, rotations):
        """
        rotations: dict with keys 'fl','rl','fr','rr' for wheel rotations
        """
        # Convert rotations to distances
        s_fl = rotations["fl"] * self.wheel_circumference
        s_rl = rotations["rl"] * self.wheel_circumference
        s_fr = rotations["fr"] * self.wheel_circumference
        s_rr = rotations["rr"] * self.wheel_circumference

        # Average left and right sides
        s_left = (s_fl + s_rl) / 2.0
        s_right = (s_fr + s_rr) / 2.0

        # Differential-drive odometry equations
        delta_theta = (s_right - s_left) / self.baseline
        delta_s = (s_left + s_right) / 2.0

        dx = delta_s * math.cos(self.heading + delta_theta / 2.0)
        dy = delta_s * math.sin(self.heading + delta_theta / 2.0)

        # Update state
        self.x += dx
        self.y += dy
        self.heading += delta_theta

        return delta_s, (self.x, self.y), math.degrees(self.heading) % 360

    def position(self):
        return self.x, self.y, math.degrees(self.heading) % 360


if __name__ == "__main__":
    robot = RobotOdometer(WHEEL_DIAMETER, BASELINE)

    distance, pos, heading = robot.travel(ROTATIONS)
    print(f"Moved {distance:.2f} m | Position: ({pos[0]:.2f}, {pos[1]:.2f}) | Heading: {heading:.1f}°")
