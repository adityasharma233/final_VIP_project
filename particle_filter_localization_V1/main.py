import pygame
import sys
import math
import time

from environment import Environment
from robot import Robot
from particle_filter import ParticleFilter
from visualization import Visualization

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Particle Filter Localization")
    clock = pygame.time.Clock()

    env = Environment(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    env.add_wall(((50, 50), (750, 50)))
    env.add_wall(((50, 50), (50, 550)))
    env.add_wall(((750, 50), (750, 550)))
    env.add_wall(((50, 550), (750, 550)))
    env.add_wall(((300, 200), (500, 200)))
    env.add_wall(((300, 400), (500, 400)))

    robot = Robot(x=100, y=100, theta=0.0, wheel_base=20, sensor_range=250, env=env)
    
    particle_filter = ParticleFilter(num_particles=1000, env=env, sensor_range=robot.sensor_range)
    
    viz = Visualization(screen=screen, env=env, robot=robot, particle_filter=particle_filter)
    
    commands = [
        (1.0, 0.0) for _ in range(100)
    ] + [
        (1.0, 0.1) for _ in range(50)
    ] + [
        (1.0, 0.0) for _ in range(100)
    ] + [
        (1.0, -0.1) for _ in range(50)
    ] 
    
    cmd_index = 0
    running = True
    last_time = time.time()

    while running:
        dt = time.time() - last_time
        last_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if cmd_index < len(commands):
            v, w = commands[cmd_index]
            cmd_index += 1
        else:
            v, w = 0.0, 0.0

        robot.move(v, w, dt)

        particle_filter.motion_update(v, w, dt)

        measurements = robot.get_sensor_readings()

        particle_filter.sensor_update(measurements)

        particle_filter.resample()

        screen.fill((255, 255, 255))
        viz.draw()
        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
