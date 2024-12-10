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
    pygame.display.set_caption("Particle Filter Localization with Goal")
    clock = pygame.time.Clock()

    env = Environment(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    env.add_wall(((50, 50), (750, 50)))
    env.add_wall(((50, 50), (50, 550)))
    env.add_wall(((750, 50), (750, 550)))
    env.add_wall(((50, 550), (750, 550)))
    env.add_wall(((300, 200), (500, 200)))
    env.add_wall(((300, 400), (500, 400)))

    goal = (700, 500)
    goal_radius = 20.0

    robot = Robot(x=100, y=100, theta=0.0, wheel_base=20, sensor_range=250, env=env)
    
    particle_filter = ParticleFilter(num_particles=1000, env=env, sensor_range=robot.sensor_range)
    
    viz = Visualization(screen=screen, env=env, robot=robot, particle_filter=particle_filter, goal=goal, goal_radius=goal_radius)

    running = True
    last_time = time.time()

    while running:
        dt = time.time() - last_time
        last_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dx = goal[0] - robot.x
        dy = goal[1] - robot.y
        dist_to_goal = math.sqrt(dx*dx + dy*dy)

        if dist_to_goal < goal_radius:
            v, w = 0.0, 0.0
        else:
            desired_angle = math.atan2(dy, dx)
            angle_diff = (desired_angle - robot.theta) % (2*math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2*math.pi

            w = 0.5 * angle_diff  
            v = 50.0  

            measurements = robot.get_sensor_readings()
            front_indices = range(len(measurements)//8, 3*(len(measurements)//8))  
            min_front_dist = min(measurements[i] for i in front_indices)
            
            if min_front_dist < 50:  
                v = 0.0
                while min_front_dist < 50:
                    w = 0.5 if angle_diff >= 0 else -0.5
                    robot.move(v, w, dt)  
                    measurements = robot.get_sensor_readings() 
                    min_front_dist = min(measurements[i] for i in front_indices)

        robot.move(v, w, dt)

        particle_filter.motion_update(v, w, dt)

        measurements = robot.get_sensor_readings()

        particle_filter.sensor_update(measurements)

        particle_filter.resample()

        screen.fill((255, 255, 255))
        viz.draw()
        viz.draw_hud(v, w)
        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
