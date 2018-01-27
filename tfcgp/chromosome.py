import numpy as np
import tensorflow as tf

class Node:

    def __init__(self, x, y, function, param):
        self.x = x
        self.y = y
        self.function = function
        self.param = param
        self.active = False
        self.inp = False

class Chromosome:

    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.genes = []
        self.nodes = []
        self.outputs = []

    def from_genes(self, genes, config):
        self.genes = genes
        self.nodes = []
        g = np.reshape(genes[self.nout:], (10, 4))
        inds = np.arange(self.nin, self.nin+config.cfg["num_nodes"])
        xs = np.round(inds * g[:,0])
        ys = np.round(inds * g[:,1])
        fs = np.floor(len(config.functions) * g[:,2]).astype(int)
        for i in range(self.nin):
            n = Node(0, 0, lambda x: x, 0.0)
            n.inp = True
            self.nodes += [n]
        for i in range(config.cfg["num_nodes"]):
            n = Node(xs[i], ys[i], config.functions[fs[i]], 2*g[i, 3]-1.0)
            self.nodes += [n]
        self.outputs = np.floor(genes[:self.nout] * len(self.nodes))

    def random(self, config):
        genes = np.random.rand(self.nout + 4*config.cfg["num_nodes"])
        self.from_genes(genes, config)
