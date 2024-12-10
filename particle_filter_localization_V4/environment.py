import math

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def add_wall(self, endpoints):
        self.walls.append(endpoints)

    def ray_cast(self, x, y, theta, max_range):
        end_x = x + max_range * math.cos(theta)
        end_y = y + max_range * math.sin(theta)

        closest_distance = max_range
        for wall in self.walls:
            dist = self.line_intersect_distance((x, y), (end_x, end_y), wall)
            if dist is not None and dist < closest_distance:
                closest_distance = dist
        return closest_distance

    @staticmethod
    def line_intersect_distance(p0, p1, wall):
        x1, y1 = p0
        x2, y2 = p1
        x3, y3 = wall[0]
        x4, y4 = wall[1]

        denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
        if denom == 0:
            return None

        ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom
        ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            ix = x1 + ua*(x2 - x1)
            iy = y1 + ua*(y2 - y1)
            dist = math.sqrt((ix - x1)**2 + (iy - y1)**2)
            return dist
        return None
