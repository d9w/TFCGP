from tfcgp.problem import Problem
from tfcgp.chromosome import Chromosome
import numpy as np
import os

class GA:

    def __init__(self, problem, config, logname='test', root_dir='.'):
        self.config = config
        self.problem = problem
        self.max_fit = 0.0
        self.population = []
        for i in range(self.config.cfg["ga_population"]):
            ch = Chromosome(self.problem.nin, self.problem.nout)
            ch.random(config)
            self.population += [ch]
        self.fits = -np.inf*np.ones(self.config.cfg["ga_population"])
        self.logfile = os.path.join(root_dir, 'logs', logname+'.log')
        self.logname = logname
        self.generation = 0

    def run(self, n_steps):
        for i in range(n_steps):
            self.step()

    def mutate(self, chromosome):
        child_genes = np.copy(chromosome.genes)
        change = np.random.rand(len(child_genes)) < self.config.cfg["mutation_rate"]
        child_genes[change] = np.random.rand(np.sum(change))
        child = Chromosome(self.problem.nin, self.problem.nout)
        child.from_genes(child_genes, self.config)
        return child

    def select(self):
        # return the winner of a random three-way tournament
        inds = np.arange(len(self.population))
        np.random.shuffle(inds)
        winner = inds[np.argmax(self.fits[inds[:3]])]
        return winner

    def step(self):
        for i in range(len(self.population)):
            if self.fits[i] == -np.inf:
                fitness, history = self.problem.get_fitness(self.population[i])
                self.fits[i] = fitness

        if np.max(self.fits) > self.max_fit:
            self.max_fit = np.max(self.fits)
            with open(self.logfile, 'a') as f:
                f.write('E,%s,%d,%d,%d,%d,%0.10f,%0.10f,%0.10f\n' %
                    (self.logname, self.generation, self.problem.eval_count, 0,
                        self.problem.epochs, 0.0, 0.0, self.max_fit))

        self.generation += 1

        new_pop = []
        new_fits = -np.inf*np.ones(self.config.cfg["ga_population"])

        n_elites = int(round(self.config.cfg["ga_population"]*
                             self.config.cfg["ga_elitism"]))
        elites = np.argsort(self.fits)[::-1]
        for i in range(n_elites):
            new_pop += [self.population[elites[i]]]
            new_fits[i] = self.fits[elites[i]]

        n_mutate = int(round(self.config.cfg["ga_population"]*
                             self.config.cfg["ga_mutation"]))
        for i in range(n_mutate):
            pid = self.select()
            child = self.mutate(self.population[pid])
            new_pop += [child]
            new_fits[n_elites+i] = -np.inf

        n_rest = self.config.cfg["ga_population"] - n_elites - n_mutate

        for i in range(n_elites + n_mutate, self.config.cfg["ga_population"]):
            pid = self.select()
            new_pop += [self.population[pid]]
            new_fits[i] = self.fits[pid]

        self.population = new_pop
        self.fits = new_fits
