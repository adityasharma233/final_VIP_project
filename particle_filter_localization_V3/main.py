import pygame
import sys
import math
import time
import numpy as np

from environment import Environment
from robot import Robot
from particle_filter import ParticleFilter
from visualization import Visualization
from dqn_agent import DQNAgent

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

GOAL = (700, 500)
GOAL_RADIUS = 20.0
COLLISION_PENALTY = -1.0
GOAL_REWARD = 10.0
STEP_COST = -0.01

ACTIONS = [
    (50.0, 0.0),    # forward
    (0.0, 1.0),     # turn left
    (0.0, -1.0),    # turn right
    (0.0, 0.0)      # stay still
]

def compute_state(robot, goal):
    dx = goal[0] - robot.x
    dy = goal[1] - robot.y
    dist_to_goal = math.sqrt(dx*dx + dy*dy)
    angle_to_goal = math.atan2(dy, dx) - robot.theta
    angle_to_goal = (angle_to_goal + math.pi) % (2*math.pi) - math.pi

    measurements = robot.get_sensor_readings()
    normalized_readings = [m / robot.sensor_range for m in measurements]

    dist_norm = dist_to_goal / (math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2))
    angle_norm = angle_to_goal / math.pi
    x_norm = robot.x / SCREEN_WIDTH
    y_norm = robot.y / SCREEN_HEIGHT

    state = [dist_norm, angle_norm, x_norm, y_norm] + normalized_readings
    return np.array(state, dtype=np.float32)

def check_collision(robot, env):
    robot_radius = 8
    for wall in env.walls:
        if line_segment_distance_to_point(wall[0], wall[1], (robot.x, robot.y)) < robot_radius:
            return True
    return False

def line_segment_distance_to_point(p1, p2, p):
    (x1, y1), (x2, y2) = p1, p2
    (x0, y0) = p
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    t = ((x0 - x1)*dx + (y0 - y1)*dy)/(dx*dx + dy*dy)
    t = max(0, min(1, t))
    x_closest = x1 + t*dx
    y_closest = y1 + t*dy
    return math.sqrt((x0 - x_closest)**2 + (y0 - y_closest)**2)

def reset_robot(robot):
    robot.x = 100
    robot.y = 100
    robot.theta = 0.0

def reached_goal(robot, goal, radius):
    dx = goal[0] - robot.x
    dy = goal[1] - robot.y
    dist = math.sqrt(dx*dx + dy*dy)
    return dist < radius

def draw_best_path(screen, best_path):
    if len(best_path) > 1:
        pygame.draw.lines(screen, (0, 0, 255), False, best_path, 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("DQN Robot Navigation")

    env = Environment(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    env.add_wall(((50, 50), (750, 50)))
    env.add_wall(((50, 50), (50, 550)))
    env.add_wall(((750, 50), (750, 550)))
    env.add_wall(((50, 550), (750, 550)))
    env.add_wall(((300, 200), (500, 200)))
    env.add_wall(((300, 400), (500, 400)))

    robot = Robot(x=100, y=100, theta=0.0, wheel_base=20, sensor_range=250, env=env)
    particle_filter = ParticleFilter(num_particles=1000, env=env, sensor_range=robot.sensor_range)
    viz = Visualization(screen=screen, env=env, robot=robot, particle_filter=particle_filter, goal=GOAL, goal_radius=GOAL_RADIUS)

    state_dim = 40
    action_dim = len(ACTIONS)
    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 200
    max_steps = 500
    target_update_freq = 10

    best_reward = float('-inf')
    best_path = []

    for episode in range(num_episodes):
        start_time = time.time()
        reset_robot(robot)
        state = compute_state(robot, GOAL)
        episode_reward = 0
        episode_path = [(robot.x, robot.y)]

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state)
            v, w = ACTIONS[action]

            dt = 0.05
            robot.move(v, w, dt)

            particle_filter.motion_update(v, w, dt)
            measurements = robot.get_sensor_readings()
            particle_filter.sensor_update(measurements)
            particle_filter.resample()

            next_state = compute_state(robot, GOAL)
            done = False
            reward = STEP_COST

            if check_collision(robot, env):
                reward = COLLISION_PENALTY
                done = True
            elif reached_goal(robot, GOAL, GOAL_RADIUS):
                reward = GOAL_REWARD
                done = True
            else:
                prev_dist = state[0]*math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
                curr_dist = next_state[0]*math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
                if curr_dist < prev_dist:
                    reward += 0.05

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            if step % target_update_freq == 0:
                agent.update_target_network()

            state = next_state
            episode_reward += reward
            episode_path.append((robot.x, robot.y))

            screen.fill((255, 255, 255))
            
            draw_best_path(screen, best_path)
            
            viz.draw()
            viz.draw_hud(v, w)
            pygame.display.flip()

            if done:
                break

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = episode_path[:]

        episode_time = time.time() - start_time
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Time: {episode_time:.2f}s")

    pygame.quit()

if __name__ == "__main__":
    main()
