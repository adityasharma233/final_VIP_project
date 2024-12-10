import pygame
import math

class Visualization:
    def __init__(self, screen, env, robot, particle_filter):
        self.screen = screen
        self.env = env
        self.robot = robot
        self.pf = particle_filter

    def draw(self):
        self.draw_walls()
        self.draw_particles()
        self.draw_robot()
        self.draw_sensor_rays()

    def draw_walls(self):
        for wall in self.env.walls:
            pygame.draw.line(self.screen, (0,0,0), wall[0], wall[1], 2)

    def draw_particles(self):
        max_w = max(self.pf.weights) if len(self.pf.weights) > 0 else 1
        for (x, y, theta), w in zip(self.pf.particles, self.pf.weights):
            intensity = int((w/max_w)*255)
            pygame.draw.circle(self.screen, (intensity, 0, 0), (int(x), int(y)), 2)

    def draw_robot(self):
        pygame.draw.circle(self.screen, (0,0,255), (int(self.robot.x), int(self.robot.y)), 8)
        end_x = self.robot.x + 20*math.cos(self.robot.theta)
        end_y = self.robot.y + 20*math.sin(self.robot.theta)
        pygame.draw.line(self.screen, (0,0,255), (self.robot.x,self.robot.y), (end_x,end_y), 2)

    def draw_sensor_rays(self):
        measurements = self.robot.get_sensor_readings()
        resolution = len(measurements)
        for i, dist in enumerate(measurements):
            angle = self.robot.theta + 2*math.pi*i/resolution
            end_x = self.robot.x + dist*math.cos(angle)
            end_y = self.robot.y + dist*math.sin(angle)
            pygame.draw.line(self.screen, (0,255,0), (self.robot.x, self.robot.y), (end_x, end_y), 1)
