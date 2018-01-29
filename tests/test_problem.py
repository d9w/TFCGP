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

# def test_run():
#     ch = Chromosome(2, 1)
#     ch.random(c)
#     ch.outputs[0] = 2
#     ch.nodes[2].x = 0
#     ch.nodes[2].y = 1
#     ch.nodes[2].function = tf.square
#     ch.nodes[2].arity = 1
#     p = Problem()
#     outv = p.run(ch)
#     print(outv)
#     assert outv == ch.nodes[2].param

# def test_multiout_run():
#     ch = Chromosome(10, 10)
#     ch.random(c)
#     p = Problem()
#     outv = p.run(ch)
#     print(outv)
#     assert len(outv) > 1

def test_problem_eval():
    p = Problem()
    ch = Chromosome(p.nin, p.nout)
    ch.random(c)
    loss = p.run(ch)
    print("loss: ", loss)
    assert True
