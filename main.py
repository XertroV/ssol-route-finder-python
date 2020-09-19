import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
import os
from pathlib import Path

import requests
import json
from bs4 import BeautifulSoup

from assol_map import coords, path_between, cities, starting_point, maze, draw_path_and_map, draw_gene_and_map, \
    calc_path_between, set_path_between

from genetic_algo import GeneticAlgo

START_TIME = int(time.time())

out_folder = Path(f"gene-out-{START_TIME}")

if not os.path.exists(out_folder):
    os.mkdir(out_folder)


# for idx_1 in range(0, len(cities) - 1):
#     for idx_2 in range(idx_1 + 1, len(cities)):
#         city_a = cities[idx_1]
#         city_b = cities[idx_2]
#         # path = path_between(city_a, city_b)
#         # dist = len(path)
#         # dist_dict[city_a][city_b] = dist
#         # path_dict[city_a][city_b] = path
#         # edges.append((city_a, city_b, dist))


try:
    g = GeneticAlgo(start=repr(starting_point), mutation_prob=0.25, crossover_prob=0.25,
                    population_size=30, steps=15, iterations=2000, out_path=out_folder)
    path = g.converge()
    draw_gene_and_map(path, maze, f"result-{START_TIME}.png")
except KeyboardInterrupt as e:
    print(f"exiting b/c of ctrl-c")
