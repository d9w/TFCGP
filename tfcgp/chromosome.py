import numpy as np
import tensorflow as tf
import yaml
from graphviz import Digraph

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
        self.g = tf.Graph()
        self.tf_out = None

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

    def recurse_tensor(self, node_id):
        n = self.nodes[node_id]
        # g.reuse_variables()
        if n.inp:
            # with g.as_default():
            return tf.get_variable("inp"+str(node_id))
                #     return tf.get_variable("inp"+str(node_id), dtype=tf.float32,
                #                         initializer=tf.constant(0.0))
        else:
            if n.arity == 1:
                # with g.as_default():
                    # with tf.variable_scope("program", reuse=tf.AUTO_REUSE):
                return tf.multiply(
                    tf.get_variable("p"+str(node_id)),
                                    # dtype=tf.float32,
                                    # initializer=tf.constant(n.param)),
                    n.function(self.recurse_tensor(n.x)))
            else:
                # with g.as_default():
                    # with tf.variable_scope("program", reuse=tf.AUTO_REUSE):
                return tf.multiply(
                    # tf.get_variable("p"+str(node_id),
                    tf.get_variable("p"+str(node_id)),
                                    # dtype=tf.float32,
                                    # initializer=tf.constant(n.param)),
                    n.function(self.recurse_tensor(n.x),
                                self.recurse_tensor(n.y)))

    def get_tensors(self):
        if self.tf_out:
            return self.tf_out
        else:
            self.set_active()
            with self.g.as_default():
                with tf.variable_scope("program"):
                    for i in range(self.nin):
                        tf.get_variable("inp"+str(i), dtype=tf.float32,
                                        initializer=tf.constant(1.0))
                    for i in range(len(self.nodes)):
                        if self.nodes[i].active and not self.nodes[i].inp:
                            tf.get_variable("p"+str(i), dtype=tf.float32,
                                            initializer=tf.constant(self.nodes[i].param))
            with self.g.as_default():
                with tf.variable_scope("program", reuse=True):
                    tf_outputs = []
                    for i in range(len(self.outputs)):
                        tf_outputs += [self.recurse_tensor(self.outputs[i])]
                    self.tf_out = tf.stack(tf_outputs)
            return self.tf_out

    def run(self):
        with self.g.as_default():
            out = self.get_tensors()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                return sess.run(out)

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
        self.outputs = np.floor(genes[:self.nout] * len(self.nodes)).astype(int)

    def random(self, config):
        genes = np.random.rand(self.nout + 4*config.cfg["num_nodes"])
        self.from_genes(genes, config)
