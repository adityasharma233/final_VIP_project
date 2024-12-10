import math
import random
from utils import normalize_weights

class ParticleFilter:
    def __init__(self, num_particles, env, sensor_range):
        self.num_particles = num_particles
        self.env = env
        self.sensor_range = sensor_range
        self.particles = []
        self.weights = []
        self.init_particles()

    def init_particles(self):
        self.particles = []
        self.weights = []
        for _ in range(self.num_particles):
            x = random.uniform(0, self.env.width)
            y = random.uniform(0, self.env.height)
            theta = random.uniform(0, 2*math.pi)
            self.particles.append((x, y, theta))
            self.weights.append(1.0/self.num_particles)

    def motion_update(self, v, w, dt):
        v_noisy = v + random.gauss(0, 0.1)
        w_noisy = w + random.gauss(0, 0.05)

        new_particles = []
        for (x, y, theta) in self.particles:
            x_new = x + v_noisy * math.cos(theta) * dt
            y_new = y + v_noisy * math.sin(theta) * dt
            theta_new = (theta + w_noisy * dt) % (2*math.pi)
            if 0 < x_new < self.env.width and 0 < y_new < self.env.height:
                new_particles.append((x_new, y_new, theta_new))
            else:
                x_new = random.uniform(0, self.env.width)
                y_new = random.uniform(0, self.env.height)
                theta_new = random.uniform(0, 2*math.pi)
                new_particles.append((x_new, y_new, theta_new))

        self.particles = new_particles

    def sensor_update(self, measurements):
        sigma = 10.0
        for i, (x, y, theta) in enumerate(self.particles):
            predicted = self.simulate_sensor(x, y, theta)
            error = sum((m - p)**2 for m, p in zip(measurements, predicted))
            likelihood = math.exp(-error/(2*sigma**2))
            self.weights[i] = likelihood

        self.weights = normalize_weights(self.weights)

    def simulate_sensor(self, x, y, theta):
        angles = self.simulate_sensor_angles(theta)
        return [self.env.ray_cast(x, y, a, self.sensor_range) for a in angles]

    def simulate_sensor_angles(self, theta):
        resolution = 36
        return [theta + 2*math.pi*i/resolution for i in range(resolution)]

    def resample(self):
        new_particles = []
        index = random.randint(0, self.num_particles-1)
        beta = 0.0
        mw = max(self.weights)
        for _ in range(self.num_particles):
            beta += random.random() * 2.0 * mw
            while beta > self.weights[index]:
                beta -= self.weights[index]
                index = (index+1) % self.num_particles
            new_particles.append(self.particles[index])
        self.particles = new_particles
        self.weights = [1.0/self.num_particles]*self.num_particles
