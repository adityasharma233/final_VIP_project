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
        self.sensor = Sensor(env=env, sensor_range=sensor_range, resolution=36) 

    

    def move(self, v, w, dt):
        speed_multiplier = 50.0  
        self.x += speed_multiplier * v * math.cos(self.theta) * dt
        self.y += speed_multiplier * v * math.sin(self.theta) * dt
        self.theta += speed_multiplier * w * dt
        self.theta %= 2 * math.pi

    def get_sensor_readings(self):
        return self.sensor.get_readings(self.x, self.y, self.theta)
