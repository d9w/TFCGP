import yaml
import tensorflow as tf

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
                        self.arity[k] = fv['arity']
                        if fk not in self.functions:
                            self.functions += [eval(fk)]
            else:
                self.cfg[k] = v

    def reset(self):
        self.functions = []
        self.arity = {}
        self.cfg = {}
