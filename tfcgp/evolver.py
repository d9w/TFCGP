from tfcgp.problem import Problem
from tfcgp.chromosome import Chromosome
import numpy as np

class Evolver:

    def __init__(self, problem, config):
        self.config = config
        self.problem = problem
        self.max_fit = 0.0
        self.best = Chromosome(self.problem.nin, self.problem.nout, self.problem.batch_size)
        self.best.random(config)
        self.eval_count = 0
        self.generation = 0

    def run(self, n_steps):
        for i in range(n_steps):
            self.step()
            # with open(self.problem.log_file, 'a') as f:
                # f.write("gen")

    def mutate(self, chromosome):
        child_genes = np.copy(chromosome.genes)
        change = np.random.rand(len(child_genes)) < self.config.cfg["mutation_rate"]
        child_genes[change] = np.random.rand(np.sum(change))
        child = Chromosome(self.problem.nin, self.problem.nout, self.problem.batch_size)
        child.from_genes(child_genes, self.config)
        return child

    def step(self):
        for i in range(self.config.cfg["lambda"]):
            child = self.mutate(self.best)
            fitness = self.problem.get_fitness(child)
            self.eval_count += 1
            if fitness >= self.max_fit:
                self.best = child
                self.max_fit = fitness
        self.generation += 1
