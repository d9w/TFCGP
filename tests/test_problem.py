from tfcgp.config import Config
from tfcgp.chromosome import Chromosome
from tfcgp.problem import Problem
import numpy as np
import tensorflow as tf
from sklearn import datasets

c = Config()
c.update("cfg/base.yaml")
data = datasets.load_iris()

def test_creation():
    p = Problem(data.data, data.target)
    print(p.x_train.shape)
    print(p.y_train.shape)
    assert True

def test_get_fitness():
    p = Problem(data.data, data.target)
    ch = Chromosome(p.nin, p.nout)
    ch.random(c)
    fitness = p.get_fitness(ch)
    print("fitness: ", fitness)
    assert True
