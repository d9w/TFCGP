from tfcgp.config import Config
from tfcgp.chromosome import Chromosome
from tfcgp.problem import Problem
import numpy as np
import tensorflow as tf

c = Config()
c.update("cfg/test.yaml")

def test_creation():
    p = Problem()
    print(p.x_train.shape)
    print(p.y_train.shape)
    assert True

def test_problem_eval():
    p = Problem()
    ch = Chromosome(p.nin, p.nout, p.batch_size)
    ch.random(c)
    outs = p.run(ch)
    print("outputs: ", outs)
    print(outs.shape)
    assert True

def test_fit():
    p = Problem()
    ch = Chromosome(p.nin, p.nout, p.batch_size)
    ch.random(c)
    loss = p.fit(ch)
    print("loss: ", loss)
    assert True


