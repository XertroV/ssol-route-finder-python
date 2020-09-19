import functools
import math
import random
from collections import defaultdict

import numpy as np
from PIL import Image


def round_pair(p):
    return round(p[0]), round(p[1])


SCALE_FACTOR = 5
starting_point = (393, 374)

raw_coords = [
    (344.61, 160.53),
    (354.08, 209.05),
    (370.42, 125.88),
    (393.14, 218.51),
    (408.9, 260.72),
    (411.37, 142.90),
    (424.62, 188.27),
    (426.53, 241.82),
    (432.88, 359.64),
    (433.47, 269.55),
    (455.49, 205.92),
    (473.23, 424.55),
    (490.85, 405.65),
    (492.65, 211.59),
    (495.27, 429.59),
    (541.12, 149.86),
    (557.49, 130.96),
    (563.8, 156.16),
    (611.06, 202.16),
    (611.12, 338.25),
    (623.67, 228.63),
    (636.89, 204.69),
    (649.55, 337.00),
    (668.54, 559.41),
    (685.54, 542.40),
    (691.85, 565.71),
    (696.16, 345.20),
    (734.44, 122.46),
    (736.48, 360.33),
    (772.45, 142.72),
    (775.68, 696.78),
    (778.05, 376.72),
    (783.14, 474.37),
    (797.67, 576.44),
    (798.95, 694.05),
    (800.2, 170.47),
    (800.77, 455.48),
    (808.34, 481.31),
    (815.22, 395.62),
    (816.69, 690.55),
    (828.71, 200.73),
    (844.06, 415.02),
    (844.19, 687.80),
    (845.53, 543.06),
    (850.72, 229.99),
    (855.66, 646.05),
    (872.18, 685.31),
    (878.72, 235.00),
    (879.81, 429.27),
    (884.14, 600.55),
    (899.43, 684.56),
    (907.77, 308.06),
    (908.47, 232.50),
    (915.38, 415.80),
    (923.49, 289.01),
    (927.42, 683.07),
    (930.75, 315.02),
    (937.21, 230.51),
    (953.65, 642.57),
    (955.29, 406.28),
    (957.16, 680.82),
    (962.01, 350.03),
    (980.45, 232.51),
    (984.84, 522.06),
    (987.16, 677.58),
    (993.53, 392.04),
    (1018.65, 669.83),
    (1028.45, 236.27),
    (1040.11, 592.83),
    (1042.77, 393.80),
    (1056.64, 656.09),
    (1067.45, 258.78),
    (1081.77, 409.56),
    (1088.62, 637.59),
    (1093.2, 270.79),
    (1118.77, 419.56),
    (1122.1, 608.59),
    (1124.56, 519.33),
    (1129.45, 278.30),
    (1161.26, 426.82),
    (1168.2, 289.31),
    (1204.07, 560.85),
    (1211.26, 436.08),
    (1246.72, 353.08),
    (1256.78, 495.35),
    (1261.65, 219.56),
    (1287.3, 554.11),
    (1287.99, 429.35),
    (1288.42, 258.57),
    (1328.67, 283.33),
    (1340.88, 202.32),
    (1356.76, 493.37),
    (1359.98, 416.61),
    (1372.04, 563.63),
    (1376.42, 309.59),
    (1388.2, 372.11),
    (1404.87, 211.34),
    (1436.91, 297.10),
    (1443.23, 443.88),
    (1497.18, 370.62),
]
random_coords = random.sample(raw_coords, 30)

USE_RANDOM = False

coords = list(map(round_pair, [starting_point] + (random_coords if USE_RANDOM else raw_coords)))

cities = list(map(repr, coords))
city_to_coords = dict(zip(cities, coords))
# print(cities)
# cities = coords


black_pixel = np.array([0, 0, 0, 255], dtype='uint8')
white_pixel = np.array([255, 255, 255, 255], dtype='uint8')

assol_raw_map = np.array(Image.open('./ASSOL_MAP_for_a_star.png'))

# print(assol_raw_map.dtype, assol_raw_map.ndim, assol_raw_map.shape)

assol_map_ = []
for row in assol_raw_map:
    row_ = []
    for pixel in row:
        row_.append(all(pixel == black_pixel))
    assol_map_.append(row_)

assol_map = np.array(assol_map_)

print(f"Generated assol_map")


def thin_2d_array_by_factor(xs, factor):
    new_arr = []
    for i, row in enumerate(xs):
        if i % factor != 0:
            continue
        new_row = []
        for j, el in enumerate(row):
            if j % factor != 0:
                continue
            new_row.append(el)
        new_arr.append(new_row)
    return np.array(new_arr)


def save_assol_map(_map: np.array):
    # print(_map.dtype, _map.ndim, _map.shape)
    assol_map_image_out = np.array([[black_pixel if p else white_pixel for p in r] for r in _map])
    # print(assol_map_image_out.dtype, assol_map_image_out.ndim, assol_map_image_out.shape)
    Image.fromarray(assol_map_image_out).save('assol-map-for-astar-output.png')


path_color = np.array([255, 19, 92, 255], dtype='uint8')


def draw_path_and_map(path, _map: np.array, filename='path-on-map.png'):
    def path_and_map_color(xy, p):
        if p:
            return black_pixel
        return white_pixel if xy not in path else path_color

    # print(_map.dtype, _map.ndim, _map.shape)
    path_and_map_out = np.array(
        [[path_and_map_color((x, y), p) for (x, p) in enumerate(r)] for (y, r) in enumerate(_map)])
    # print(assol_map_image_out.dtype, assol_map_image_out.ndim, assol_map_image_out.shape)
    Image.fromarray(path_and_map_out).save(filename)
    print(f"wrote out {filename}")


def draw_gene_and_map(gene, _map: np.array, filename):
    path = []
    for i in range(len(gene) - 1):
        # eval to get tuples
        g1 = gene[i]
        g2 = gene[i + 1]
        # print(repr((g1, g2)))
        path += path_between(g1, g2)
    return draw_path_and_map(path, _map, filename)


from astar import AStar





class ASSOL_Solver(AStar):
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, maze):
        self.lines = maze
        self.width = len(self.lines[0])
        self.height = len(self.lines)

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        return 1.4142 if n1[0] != n2[0] and n1[1] != n2[1] else 1

    def is_okay_node(self, node):
        nx, ny = node
        return 0 <= ny < self.height and 0 <= nx < self.width and not maze[ny, nx]

    def make_ns(self, node):
        x, y = node
        moves = (-1, 0, 1)
        ret = []
        for dx in moves:
            for dy in moves:
                if dx == 0 and dy == 0:
                    continue
                to_add = (x + dx, y + dy)
                nx, ny = to_add
                if not (0 <= ny < self.height and 0 <= nx < self.width and not maze[ny, nx]):
                    continue
                ret.append(to_add)
        return ret
        # return list(filter(self.is_okay_node,
        #                    [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y),
        #                     (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]))

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        ns = self.make_ns(node)
        # print(f"returning neighbours {ns}")
        return ns

    @functools.lru_cache(maxsize=5000000)
    def is_goal_reached(self, current, goal):
        # print(f"Testing if goal reached. {current} ==? {goal}")
        return current[0] == goal[0] and current[1] == goal[1]


# if __name__ == "__main__":
# maze = assol_map
#
# start = [0, 0]  # starting position
# end = [4, 5]  # ending position
# cost = 1  # cost per movement
# path = search(maze, cost, start, end)
# print(path)


known_paths = defaultdict(dict)

maze = thin_2d_array_by_factor(assol_map, SCALE_FACTOR)


# print(len(maze), len(maze[0]), maze.shape)
# print(maze[starting_point[0] // SCALE_FACTOR, starting_point[1] // SCALE_FACTOR])
# save_assol_map(maze)


@functools.lru_cache(maxsize=500000)
def path_between(c1, c2):
    p1 = city_to_coords[c1]
    p2 = city_to_coords[c2]
    # p1 = c1
    # p2 = c2
    p1 = (p1[0] // SCALE_FACTOR, p1[1] // SCALE_FACTOR)
    p2 = (p2[0] // SCALE_FACTOR, p2[1] // SCALE_FACTOR)
    if known_paths[c1].get(c2, None) is None:
        path = tuple(ASSOL_Solver(maze).astar(p1, p2))
        set_path_between(c1, c2, path)
        # print(f"Found path between {p1} and {p2} with length {len(path)}")
    else:
        path = known_paths[c1][c2]
    return path


def calc_path_between(c1, c2):
    p1 = city_to_coords[c1]
    p2 = city_to_coords[c2]
    p1 = (p1[0] // SCALE_FACTOR, p1[1] // SCALE_FACTOR)
    p2 = (p2[0] // SCALE_FACTOR, p2[1] // SCALE_FACTOR)
    path = list(ASSOL_Solver(maze).astar(p1, p2))
    return path


def set_path_between(c1, c2, path):
    known_paths[c1][c2] = path


edges = []
dist_dict = defaultdict(dict)
path_dict = defaultdict(dict)


def lookup_dist(city_a, city_b):
    if city_b not in dist_dict[city_a]:
        path = path_between(city_a, city_b)
        if path is None:
            path = calc_path_between(city_a, city_b)
            set_path_between(city_a, city_b, path)
        dist = len(path)
        dist_dict[city_a][city_b] = dist
        path_dict[city_a][city_b] = path
        edges.append((city_a, city_b, dist))
        return dist
    return dist_dict[city_a][city_b]


if __name__ == "__main__":
    p1 = repr(starting_point)
    # p1 = starting_point
    p2 = random.choice(cities)
    path = path_between(p1, p2)
    draw_path_and_map(path, maze)
