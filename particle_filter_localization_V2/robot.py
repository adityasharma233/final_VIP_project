import math
from sensor import Sensor

class Robot:
    def __init__(self, x, y, theta, wheel_base, sensor_range, env):
        self.x = x
        self.y = y
        self.theta = theta
        self.wheel_base = wheel_base
        self.sensor_range = sensor_range
        self.env = env
        self.sensor = Sensor(env=env, sensor_range=sensor_range, resolution=36)  # 36 rays for simplicity

    def move(self, v, w, dt):
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta %= 2 * math.pi

    def get_sensor_readings(self):
        return self.sensor.get_readings(self.x, self.y, self.theta)
