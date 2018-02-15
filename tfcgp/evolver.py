from tfcgp.problem import Problem
from tfcgp.chromosome import Chromosome
import numpy as np
import os

class Evolver:

    def __init__(self, problem, config, logname='test', root_dir='.'):
        self.config = config
        self.problem = problem
        self.max_fit = 0.0
        self.best = Chromosome(self.problem.nin, self.problem.nout)
        self.best.random(config)
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

    def step(self):
        next_best = self.best
        next_max_fit = self.max_fit
        for i in range(self.config.cfg["lambda"]):
            child = self.mutate(self.best)
            fitness, history = self.problem.get_fitness(child)
            if fitness >= self.max_fit:
                if fitness > self.max_fit:
                    with open(self.logfile, 'a') as f:
                        for i in range(len(history)):
                            # type,logname,gen,eval,epoch,total_epochs,loss,acc,best
                            f.write('L,%s,%d,%d,%d,%d,%0.10f,%0.10f,%0.10f\n' %
                                (self.logname, self.generation, self.problem.eval_count, i,
                                 self.problem.epochs, history[i][0], history[i][1], fitness))
                        f.write('E,%s,%d,%d,%d,%d,%0.10f,%0.10f,%0.10f\n' %
                            (self.logname, self.generation, self.problem.eval_count, 0,
                                self.problem.epochs, 0.0, 0.0, fitness))
                next_best = child
                next_max_fit = fitness
            else:
                del child
        self.best = next_best
        self.max_fit = next_max_fit
        self.generation += 1
