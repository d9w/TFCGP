import numpy as np
import tensorflow as tf
import yaml
from graphviz import Digraph
from . import config as custom

class Node:

    def __init__(self, x, y, function, arity, param):
        self.x = x
        self.y = y
        self.function = function
        self.param = param
        self.arity = arity
        self.active = False
        self.inp = False

    def __str__(self):
        return str({'x': self.x,
                    'y': self.y,
                    'f': self.function.__name__,
                    'p': self.param,
                    'arity': self.arity,
                    'active': self.active,
                    'inp': self.inp})

class Chromosome:

    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.genes = []
        self.nodes = []
        self.outputs = []
        self.tf_out = None
        self.processed = False

    def recurse_active(self, node_id):
        n = self.nodes[node_id]
        if not n.active:
            n.active = True
            if not n.inp:
                self.recurse_active(n.x)
                if n.arity == 2:
                    self.recurse_active(n.y)

    def set_active(self):
        for output in self.outputs:
            self.recurse_active(output)

    def get_active(self):
        return list(np.where([n.active for n in self.nodes])[0])

    def set_params(self, p):
        for k, v in self.param_id.items():
            self.nodes[k].param = p[v]
            self.genes[self.nout+4*(k-self.nin)+3] = (p[v]+1.0)/2.0

    def setup(self):
        self.set_active()
        self.params = []
        self.param_id = {}
        for i in range(len(self.nodes)):
            if self.nodes[i].active and not self.nodes[i].inp:
                self.param_id[i] = len(self.params)
                self.params.append(
                    tf.maximum(
                        tf.constant(-1.0),
                        tf.minimum(
                            tf.constant(1.0),
                            tf.get_variable("p"+str(i), dtype=tf.float32,
                                            initializer=tf.constant(self.nodes[i].param)))))

    def recurse_tensor(self, node_id, inputs):
        n = self.nodes[node_id]
        if n.inp:
            inp = tf.to_float(tf.unstack(inputs, axis=1)[node_id])
            return inp
        else:
            if n.arity == 1:
                return tf.multiply(
                    self.params[self.param_id[node_id]],
                    n.function(self.recurse_tensor(n.x, inputs)))
            else:
                return tf.multiply(
                    self.params[self.param_id[node_id]],
                    n.function(self.recurse_tensor(n.x, inputs),
                                self.recurse_tensor(n.y, inputs)))

    def get_tensors(self, inputs):
        tf_outputs = []
        for i in range(len(self.outputs)):
            # tf_outputs += [tf.reduce_mean(tf.reduce_mean(
                # self.recurse_tensor(self.outputs[i], inputs)))]
            out = self.recurse_tensor(self.outputs[i], inputs)
            if len(out.shape) > 1:
                out = tf.reduce_mean(out, axis=1)
            # if len(out.shape) == 0:
            #     out = tf.tile(out, tf.constant(inputs.shape[0]))
            tf_outputs += [out]
            # tf_outputs += [tf.reduce_mean(
            #     self.recurse_tensor(self.outputs[i], inputs),
            #     axis=1)]
        self.tf_out = tf.transpose(tf.stack(tf_outputs, axis=0))
        self.tf_out = tf.where(tf.is_nan(self.tf_out),
                               tf.zeros_like(self.tf_out), self.tf_out)
        self.processed = True
        return self.tf_out

    def visualize(self, filename):
        _ = self.get_tensors()
        dot = Digraph()
        for n in self.g.as_graph_def().node:
            dot.node(n.name, label=n.name)
            for i in n.input:
                dot.edge(i, n.name)
        dot.render(filename=filename)

    def from_genes(self, genes, config):
        self.genes = genes
        self.nodes = []
        g = np.reshape(genes[self.nout:], (-1, 4))
        inds = np.arange(self.nin, self.nin+config.cfg["num_nodes"])
        xs = np.floor(inds * g[:,0]).astype(int)
        ys = np.floor(inds * g[:,1]).astype(int)
        fs = np.floor(len(config.functions) * g[:,2]).astype(int)
        for i in range(self.nin):
            n = Node(i, i, lambda x: x, 1, 0.0)
            n.inp = True
            self.nodes += [n]
        for i in range(config.cfg["num_nodes"]):
            f = config.functions[fs[i]]
            n = Node(xs[i], ys[i], eval(f), config.arity[f], float(2*g[i, 3]-1.0))
            self.nodes += [n]
        self.outputs = (np.floor(genes[:self.nout] * (len(self.nodes)-self.nin))
                        +self.nin).astype(int)
        self.processed = False

    def random(self, config):
        genes = np.random.rand(self.nout + 4*config.cfg["num_nodes"])
        self.from_genes(genes, config)
