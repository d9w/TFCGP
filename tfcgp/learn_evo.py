from tfcgp.problem import Problem
from tfcgp.chromosome import Chromosome
import numpy as np
import os

class LearnEvolver:

    def __init__(self, problem, config, logname='test', root_dir='.'):
        self.config = config
        self.problem = problem
        self.epochs = 1*self.problem.epochs
        self.max_learn_fit = 0.0
        self.max_evo_fit = 0.0
        self.max_fit = 0.0
        self.evo_best = Chromosome(self.problem.nin, self.problem.nout)
        self.evo_best.random(config)
        self.learn_best = Chromosome(self.problem.nin, self.problem.nout)
        self.learn_best.from_genes(self.evo_best.genes, config)
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
        evo_delete = True
        learn_delete = True
        evo_child = self.mutate(self.evo_best)
        self.problem.epochs = 0
        fitness, history = self.problem.get_fitness(evo_child)
        if fitness >= self.max_evo_fit:
            self.evo_best = evo_child
            self.max_evo_fit = fitness
            evo_delete = False
        if fitness >= self.max_learn_fit:
            self.learn_best = evo_child
            self.max_learn_fit = fitness
            evo_delete = False
        learn_child = self.mutate(self.learn_best)
        self.problem.epochs = self.epochs
        fitness, history = self.problem.get_fitness(learn_child)
        if fitness >= self.max_learn_fit:
            self.learn_best = learn_child
            self.max_learn_fit = fitness
            learn_delete = False
        new_max = max(self.max_evo_fit, self.max_learn_fit)
        if new_max > self.max_fit:
            self.max_fit = new_max
            with open(self.logfile, 'a') as f:
                for i in range(len(history)):
                    # type,logname,gen,eval,epoch,total_epochs,loss,acc,best
                    f.write('L,%s,%d,%d,%d,%d,%0.10f,%0.10f,%0.10f\n' %
                        (self.logname, self.generation, self.problem.eval_count, i,
                            self.problem.epochs, history[i][0], history[i][1], new_max))
                f.write('E,%s,%d,%d,%d,%d,%0.10f,%0.10f,%0.10f\n' %
                    (self.logname, self.generation, self.problem.eval_count, 0,
                        self.problem.epochs, 0.0, 0.0, new_max))
        if evo_delete:
            del evo_child
        if learn_delete:
            del learn_child
        self.generation += 1
