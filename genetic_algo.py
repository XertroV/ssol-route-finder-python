#
# I *think* I found this algorithm here: https://gist.github.com/jdmoore7/de13d106b9bb92fe111ba14224341045
# I've modified it a bit, but it's mostly the same.
#


import sys
from datetime import datetime
import random
import operator
from multiprocessing import Pool
from pathlib import Path
import threading

from numpy import vectorize
import functools

from assol_map import cities, draw_gene_and_map, maze, lookup_dist, dist_dict


@functools.lru_cache(maxsize=500000)
def get_fitness(gene):
    total_distance = 0
    for idx in range(1, len(gene)):
        city_b = gene[idx]
        city_a = gene[idx - 1]
        try:
            if city_b not in dist_dict[city_a]:
                dist = lookup_dist(city_a, city_b)
            else:
                dist = dist_dict[city_a][city_b]
        except KeyboardInterrupt as e:
            return sys.exit()
        except BrokenPipeError as e:
            return sys.exit()
        total_distance += dist
    fitness = 1 / total_distance
    return fitness


class GeneticAlgo:
    def __init__(self, start, steps=2, crossover_prob=0.15, mutation_prob=0.15, population_size=5,
                 iterations=100, out_path=Path("gene-out")):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population_size = population_size
        self.steps = steps
        self.iterations = iterations
        self.start = start
        self.cities = list([k for k in cities])
        self.cities.remove(start)
        self.genes = []
        self.epsilon = 1 - 1 / self.iterations
        self.generate_genes = vectorize(self.generate_genes)
        self.evaluate_fitness = vectorize(self.evaluate_fitness)
        self.evolve = vectorize(self.evolve)
        self.prune_genes = vectorize(self.prune_genes)
        self.converge = vectorize(self.converge)
        self.out_path = out_path

        self.generate_genes()

    def generate_genes(self):
        self.genes.append(('(393, 374)', '(433, 360)', '(427, 242)', '(491, 406)', '(624, 229)', '(1376, 310)',
                           '(1028, 236)', '(1043, 394)', '(1204, 561)', '(954, 643)', '(798, 576)', '(669, 559)',
                           '(1329, 283)', '(1288, 259)', '(1360, 417)', '(962, 350)', '(955, 406)', '(1341, 202)',
                           '(980, 233)', '(829, 201)', '(879, 235)', '(937, 231)', '(801, 455)', '(908, 308)',
                           '(1211, 436)', '(686, 542)', '(844, 688)', '(846, 543)', '(987, 678)', '(1257, 495)',
                           '(1497, 371)', '(1287, 554)', '(1443, 444)', '(1168, 289)', '(908, 232)', '(994, 392)',
                           '(1057, 656)', '(1357, 493)', '(495, 430)', '(736, 360)', '(696, 345)', '(493, 212)',
                           '(872, 685)', '(783, 474)', '(957, 681)', '(1082, 410)', '(931, 315)', '(1388, 372)',
                           '(1122, 609)', '(1119, 420)', '(1288, 429)', '(884, 601)', '(800, 170)', '(1247, 353)',
                           '(1093, 271)', '(1437, 297)', '(1262, 220)', '(1372, 564)', '(776, 697)', '(927, 683)',
                           '(915, 416)', '(345, 161)', '(778, 377)', '(637, 205)', '(541, 150)', '(808, 481)',
                           '(1089, 638)', '(899, 685)', '(433, 270)', '(856, 646)', '(692, 566)', '(411, 143)',
                           '(557, 131)', '(425, 188)', '(354, 209)', '(370, 126)', '(564, 156)', '(611, 202)',
                           '(1125, 519)', '(1129, 278)', '(844, 415)', '(1405, 211)', '(799, 694)', '(1019, 670)',
                           '(817, 691)', '(473, 425)', '(650, 337)', '(880, 429)', '(815, 396)', '(611, 338)',
                           '(409, 261)', '(923, 289)', '(1067, 259)', '(851, 230)', '(734, 122)', '(772, 143)',
                           '(1040, 593)', '(985, 522)', '(1161, 427)'))
        for c in self.cities:
            if c not in self.genes[0]:
                print(c)
                self.genes[0] = tuple(list(self.genes[0]) + [c])
        self.genes = []
        for i in range(self.population_size - len(self.genes)):
            gene = [self.start]
            options = [k for k in self.cities]
            while len(gene) < len(self.cities) + 1:
                city = random.choice(options)
                loc = options.index(city)
                gene.append(city)
                del options[loc]
            # don't end back at the start.
            # gene.append(self.start)
            self.genes.append(tuple(gene))
        # print(list((len(g), len(cities)) for g in self.genes))
        assert all([len(cities) == len(g) for g in self.genes])
        return self.genes

    def evaluate_fitness(self):
        fitness_scores = []
        DO_MULTIPROCESS = False
        try:
            if DO_MULTIPROCESS:
                with Pool(8) as p:
                    # for fitness_res in p.starmap(get_fitness, zip(self.genes, [dist_dict for g in self.genes])):
                    for fitness_res in p.map(get_fitness, self.genes):
                        fitness_scores.append(fitness_res)
            else:
                for gene in self.genes:
                    fitness_scores.append(get_fitness(gene))
            return fitness_scores
        except KeyboardInterrupt as e:
            return sys.exit()

    def evolve(self):
        index_map = {i: '' for i in range(1, len(self.cities) - 1)}
        indices = [i for i in range(1, len(self.cities) - 1)]
        to_visit = [c for c in self.cities]
        cross = (1 - self.epsilon) * self.crossover_prob
        mutate = self.epsilon * self.mutation_prob
        crossed_count = int(cross * len(self.cities) - 1)
        mutated_count = int((mutate * len(self.cities) - 1) / 2)
        for idx in range(len(self.genes) - 1):
            gene = self.genes[idx]
            for i in range(crossed_count):
                try:
                    gene_index = random.choice(indices)
                    sample = gene[gene_index]
                    if sample in to_visit:
                        index_map[gene_index] = sample
                        loc = indices.index(gene_index)
                        del indices[loc]
                        loc = to_visit.index(sample)
                        del to_visit[loc]
                    else:
                        continue
                except:
                    pass
        last_gene = self.genes[-1]
        remaining_cities = [c for c in last_gene if c in to_visit]
        for k, v in index_map.items():
            if v != '':
                continue
            else:
                city = remaining_cities.pop(0)
                index_map[k] = city
        new_gene = [index_map[i] for i in range(1, len(self.cities) - 1)]
        new_gene.insert(0, self.start)
        # don't want to end at the start
        # new_gene.append(self.start)
        for i in range(mutated_count):
            choices = [c for c in new_gene if c != self.start]
            city_a = random.choice(choices)
            city_b = random.choice(choices)
            index_a = new_gene.index(city_a)
            index_b = new_gene.index(city_b)
            new_gene[index_a] = city_b
            new_gene[index_b] = city_a
        self.genes.append(tuple(new_gene))

    def prune_genes(self):
        for i in range(self.steps):
            self.evolve()
        fitness_scores = self.evaluate_fitness()
        for i in range(self.steps):
            worst_gene_index = fitness_scores.index(min(fitness_scores))
            del self.genes[worst_gene_index]
            del fitness_scores[worst_gene_index]
        return max(fitness_scores), self.genes[fitness_scores.index(max(fitness_scores))]

    def converge(self):
        current_best_gene = Exception('no best gene ever found.')

        for i in range(self.iterations):
            print(f"[{i:4d}] -- {datetime.now().isoformat()} -- starting iteration.")
            values = self.prune_genes()
            current_score = values[0]
            current_best_gene = values[1]
            self.epsilon -= 1 / self.iterations
            # if i % 100 == 0 or True:
            # print(f"{datetime.now().isoformat()} -- {int(1 / current_score)} px")
            print(f"[{i:4d}] -- {datetime.now().isoformat()} -- {int(1 / current_score)} px")
            # print(current_best_gene)
            with open('best-assol-route-thus-far.txt', 'a') as f:
                f.write(f"\n[{i:4d}] -- {datetime.now().isoformat()} -- {current_best_gene}\n")
            if self.out_path:
                t = threading.Thread(target=lambda: draw_gene_and_map(current_best_gene, maze, filename=self.out_path / f"path-iter-{i}.png"), args=())
                # draw_gene_and_map(current_best_gene, maze, filename=self.out_path / f"path-iter-{i}.png")
                t.daemon = True
                t.start()

        return current_best_gene
