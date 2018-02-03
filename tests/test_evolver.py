from tfcgp.problem import Problem
from tfcgp.config import Config
from tfcgp.evolver import Evolver
from sklearn import datasets
import numpy as np

c = Config()
c.update("cfg/test.yaml")
data = datasets.load_iris()
p = Problem(data.data, data.target)

def test_creation():
    e = Evolver(p, c)
    assert e.max_fit == 0.0

def test_mutate():
    e = Evolver(p, c)
    child = e.mutate(e.best)
    assert np.any(child.genes != e.best.genes)

def test_improvement():
    e = Evolver(p, c)
    e.run(10)
    assert e.max_fit > 0.0

def test_lamarckian():
    p = Problem(data.data, data.target, lamarckian=True)
    e = Evolver(p, c)
    e.run(10)
    assert e.max_fit > 0.0
