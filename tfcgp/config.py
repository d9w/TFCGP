import yaml
import tensorflow as tf

def first(x):
    if len(x.shape) > 1:
        return x[:, 0]
    return x

def last(x):
    if len(x.shape) > 1:
        return x[:, -1]
    return x

def y_index(x, y):
    if len(x.shape) > 1:
        index = y
        if len(y.shape) > 1:
            index = tf.reduce_mean(y, axis=1)
        index = tf.floor(tf.multiply(y, x.shape[2]))
        return x[:, index]
    return x

def reduce_max(x):
    if len(x.shape) > 1:
        return tf.reduce_max(x, axis=1)
    return x

def reduce_min(x):
    if len(x.shape) > 1:
        return tf.reduce_min(x, axis=1)
    return x

def reduce_mean(x):
    if len(x.shape) > 1:
        return tf.reduce_mean(x, axis=1)
    return x

def reduce_sum(x):
    if len(x.shape) > 1:
        return tf.reduce_sum(x, axis=1)
    return x

def reduce_prod(x):
    if len(x.shape) > 1:
        return tf.reduce_prod(x, axis=1)
    return x

class Config:

    def __init__(self):
        self.functions = []
        self.arity = {}
        self.cfg = {}

    def update(self, cfg_file):
        cfg = yaml.load(open(cfg_file).read())
        for k, v in cfg.items():
            if k == "functions":
                for f in v:
                    for fk, fv in f.items():
                        self.arity[fk] = fv['arity']
                        if fk not in self.functions:
                            self.functions += [fk]
            else:
                self.cfg[k] = v

    def reset(self):
        self.functions = []
        self.arity = {}
        self.cfg = {}
