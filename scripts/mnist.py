from tfcgp.problem import Problem
from tfcgp.config import Config
from tfcgp.evolver import Evolver
from tensorflow.contrib.keras import datasets
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
parser.add_argument('--config', type=str, help='Config file', default='cfg/base.yaml')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--seed', type=int, help='Random seed', default=0)
args = parser.parse_args()

train, test = datasets.mnist.load_data()
data = np.concatenate((train[0], test[0]))
targets = np.concatenate((train[1], test[1]))

c = Config()
c.update(args.config)
p = Problem(data, targets, learn=args.learn, epochs=args.epochs)
e = Evolver(p, c, logname=args.log)
while p.eval_count < c.cfg["total_evals"]:
    e.step()
