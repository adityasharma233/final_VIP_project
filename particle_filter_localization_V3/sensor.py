import math

class Sensor:
    def __init__(self, env, sensor_range, resolution=36):
        self.env = env
        self.sensor_range = sensor_range
        self.resolution = resolution

    def get_readings(self, x, y, theta):
        angles = [theta + 2*math.pi*i/self.resolution for i in range(self.resolution)]
        readings = []
        for a in angles:
            dist = self.env.ray_cast(x, y, a, self.sensor_range)
            readings.append(dist)
        return readings
