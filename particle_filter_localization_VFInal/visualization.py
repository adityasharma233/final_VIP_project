import pygame
import math

class Visualization:
    def __init__(self, screen, env, robot, particle_filter, goal, goal_radius):
        self.screen = screen
        self.env = env
        self.robot = robot
        self.pf = particle_filter
        self.goal = goal
        self.goal_radius = goal_radius
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 18)

    def draw(self):
        self.draw_walls()
        self.draw_particles()
        self.draw_robot()
        self.draw_sensor_rays()
        self.draw_goal()

    def draw_walls(self):
        for wall in self.env.walls:
            pygame.draw.line(self.screen, (0,0,0), wall[0], wall[1], 2)

    def draw_particles(self):
        if len(self.pf.weights) > 0:
            max_w = max(self.pf.weights)
        else:
            max_w = 1
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

    def draw_goal(self):
        pygame.draw.circle(self.screen, (255,0,0), (int(self.goal[0]), int(self.goal[1])), int(self.goal_radius), 2)

    def draw_legend(self, best_episode, best_reward, current_episode, current_reward, total_episodes, control_text):
        # Draw a solid gray background rectangle at bottom-left
        legend_width = 350
        legend_height = 80
        x_offset = 10
        y_offset = self.screen.get_height() - legend_height - 10
        pygame.draw.rect(self.screen, (200, 200, 200), (x_offset, y_offset, legend_width, legend_height))

        lines = []
        if best_episode is not None:
            lines.append(f"Best Episode:  {best_episode}/{total_episodes}, Reward: {best_reward:.2f}")
        else:
            lines.append(f"Best Episode:  N/A")

        lines.append(f"Current Episode: {current_episode}/{total_episodes}, Reward: {current_reward:.2f}")
        lines.append(control_text)

        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (0,0,0))
            self.screen.blit(text_surface, (x_offset + 10, y_offset + 10 + i*20))
