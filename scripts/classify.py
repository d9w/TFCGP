from tfcgp.problem import Problem
from tfcgp.config import Config
from tfcgp.evolver import Evolver
from tfcgp.learn_evo import LearnEvolver
from tfcgp.ga import GA
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='CGP with Tensorflow')
parser.add_argument('--no-learn', dest='learn', action='store_const',
                    const=False, default=True,
                    help='Turn off learning')
parser.add_argument('--no-evo', dest='evo', action='store_const',
                    const=False, default=True,
                    help='Turn off evolution')
parser.add_argument('--lamarck', dest='lamarck', action='store_const',
                    const=True, default=False,
                    help='Turn on Lamarckian evolution')
parser.add_argument('--log', type=str, help='Log file')
parser.add_argument('--data', type=str, help='Data file', default='data/glass.dt')
parser.add_argument('--config', type=str, help='Config file', default='cfg/base.yaml')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--seed', type=int, help='Random seed', default=0)
args = parser.parse_args()

data = []; targets = []
nin = 0
with open(args.data, 'r') as p:
    for i in p:
        nin = int(i.strip('\n').split(' ')[1])
        break
all_dat = np.genfromtxt(args.data, delimiter=' ', skip_header=4)
data = all_dat[:, :nin]
targets = all_dat[:, nin:]

c = Config()
c.update(args.config)
p = Problem(data, targets, learn=args.learn, epochs=args.epochs, lamarckian=args.lamarck)
e = GA(p, c, logname=args.log)
while p.eval_count < c.cfg["total_evals"]:
    e.step()
